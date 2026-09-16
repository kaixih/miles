#!/usr/bin/env python3
"""Offline retained-evidence adapter. Fresh output only; no network or GPU calls."""
import argparse
from decimal import Decimal
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
LAB = ROOT.parents[1] / 'lab/rubin_two_node'
sys.path.insert(0, str(LAB))
import analyze_sglang_graph_trace as analyzer
import sglang_graph_capture as capture
import run_sglang_graph_diagnostic as wrapper


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def load(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16*1024**2:
        raise ValueError('Missing, symlinked or oversized JSON: ' + str(path))
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=float, allow_nan=False) + '\n')


def ref(path): return {'path': str(path.resolve()), 'sha256': sha(path)}


class Retained:
    """Every consumed node artifact must match a successful retention manifest."""
    def __init__(self, root):
        self.root = root.resolve()
        self.node = self.root / 'node-output'
        self.receipt_path = self.root / 'diagnostic-retention.json'
        self.receipt = load(self.receipt_path)
        if self.receipt.get('source_before_after_and_destination_verified') is not True:
            raise ValueError('Successful retention receipt required')
        self.files = self.receipt['files']
        self.checked = {}

    def path(self, name):
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe retained path')
        path = self.node / relative
        if any(p.is_symlink() for p in [path, *path.parents]):
            raise ValueError('Retained artifact symlink')
        item = self.files.get(name)
        if not item or not path.is_file() or path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Retention size/hash mismatch: ' + name)
        self.checked[name] = item
        return path

    def read(self, name): return load(self.path(name))


def identity(store, binding, expected_capture):
    candidates = [store.root / n for n in ['recovery-plan.json', 'operator-plan.json'] if (store.root / n).is_file()]
    if len(candidates) != 1: raise ValueError('One exact operator/recovery plan required')
    plan_path = candidates[0]; host = load(plan_path); config = host['config']
    ident = store.read('diagnostic-identity.json')
    wrapper.validate_identity(ident)
    if (config['run_id'] != binding['run_id'] or config['diagnostic_run'] != expected_capture
            or ident['run_id'] != expected_capture or ident['image_reference'] != config['image']):
        raise ValueError('Main/capture/image identity mismatch')
    submission = host.get('main_submission_id', host.get('ray_job', {}).get('submission_id'))
    if not submission or submission != ident.get('main_submission_id'):
        raise ValueError('Main Ray submission identity mismatch')
    if not ident.get('initial_model_inventory') or ident.get('initial_model_id') != capture.json_sha256(ident.get('initial_model_inventory')):
        raise ValueError('Initial model inventory identity mismatch')
    plan = store.read('diagnostic/plan.json')
    if plan.get('identity') != ident: raise ValueError('Wrapper/operator identity mismatch')
    if plan.get('source_sha256') != ident.get('diagnostic_source_sha256'):
        raise ValueError('Wrapper source provenance mismatch')
    plan['host_train_sha256'] = config.get('train_sha256')
    return ident, plan, [ref(plan_path), ref(store.receipt_path)]


def request_check(record, frozen, expected_prefix):
    req = record['request']
    body = req['request']
    if not body.get('rid') or len(body['rid']) != 128: raise ValueError('Expected 128 request IDs')
    expected = capture.generation_request(frozen, expected_prefix)
    for key in ['request', 'request_sha256', 'workload_sha256', 'input_ids_sha256', 'source']:
        if req.get(key) != expected[key]: raise ValueError('Frozen request mismatch: ' + key)
    flush = record['flush']
    if flush.get('status') != 200 or not flush.get('body', '').startswith('Cache flushed.\n'):
        raise ValueError('Missing successful cold-cache flush')
    return req


def condition(store, mode, ident, frozen):
    prefix = 'diagnostic/' + mode + '/'
    if prefix + 'summary.json' not in store.files: return None, []
    summary = store.read(prefix + 'summary.json')
    if summary.get('mode') != mode or summary.get('engine_identity', {}).get('run_id') != ident['run_id']:
        raise ValueError('Mode/engine identity mismatch')
    info = store.read(prefix + 'server-info.json')
    server = json.loads(info['body']); graph = server.get('cuda_graph_config', {})
    if (info['status'] != 200 or server.get('tp_size') != 1 or server.get('pp_size') != 1
            or server.get('base_gpu_id') != 0 or server.get('model_path') != frozen['source']['tokenizer_model']
            or graph.get('decode', {}).get('backend') != ('full' if mode == 'on' else 'disabled')
            or graph.get('prefill', {}).get('backend') != 'disabled'):
        raise ValueError('Resolved graph/layout policy mismatch')
    samples, literal, workloads, contexts = [], [], set(), set()
    for i in range(1, 4):
        record = store.read(prefix + f'measured-{i}.json')
        if record.get('kind') != 'measured' or record.get('iteration') != i:
            raise ValueError('Measured sample index mismatch')
        req = request_check(record, frozen, f"{ident['run_id']}-{mode}-{i}")
        seconds = record['metrics']['generation_request_seconds']
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Invalid raw HTTP duration')
        recomputed = wrapper.batch_metrics(record['response'], 128, seconds)
        if recomputed != record['metrics']: raise ValueError('Raw response/measurement mismatch')
        samples.append(recomputed); literal.append(req['request_sha256'])
        workloads.add(req['workload_sha256']); contexts.add(req['input_ids_sha256'])
    if summary.get('measurements') != samples or len(workloads) != 1 or len(contexts) != 1:
        raise ValueError('Summary/raw samples differ')
    result = {'image_digest': ident['image_reference'].split('@', 1)[1],
        'initial_model_id': ident['initial_model_id'], 'request_sha256': next(iter(workloads)),
        'context_sha256': next(iter(contexts)), 'cache_policy': 'successful_radix_flush_before_each_batch',
        'prefill_graph': False, 'decode_graph': mode == 'on',
        'timing_scope': 'http_request_prefill_decode_queue_response',
        'generation_seconds': [s['generation_request_seconds'] for s in samples],
        'raw_measurements': samples, 'literal_request_sha256': literal,
        'request_hash_semantics': 'Canonical workload excludes only rid; literal request hashes retained above.',
        'context_hash_semantics': 'Frozen initial input IDs; does not establish matched per-forward KV lengths or routing.'}
    claim = store.read(prefix + 'capture-claim.json')
    request = store.read(prefix + 'capture-request.json')
    request_check(request, frozen, ident['run_id'] + '-' + mode + '-profile')
    wrapper.batch_metrics(request['response'], 128, 1)
    if claim.get('engine') != summary['engine_identity'] or request['arm']['status'] != 200:
        raise ValueError('Capture/engine identity mismatch')
    payload = claim['payload']
    if (payload.get('profile_id') != ident['run_id'] + '-' + mode
            or payload.get('profile_by_stage') is not True or payload.get('with_stack') is not False
            or payload.get('record_shapes') is not False or payload.get('num_steps') != 4):
        raise ValueError('Unexpected capture policy')
    analyses = []
    seen = set()
    for item in summary['capture']['trace_evidence']:
        name = Path(item['path']).name
        if name in seen or not name.startswith(payload['profile_id'] + '-'):
            raise ValueError('Duplicate/unbound trace')
        seen.add(name)
        path = store.path(prefix + 'traces/' + name)
        if sha(path) != item['sha256']: raise ValueError('Capture/retention trace SHA mismatch')
        analysis = analyzer.analyze(path, item['sha256'])
        analyses.append(analysis)
    return result, analyses


def select_forward(analyses, stage):
    candidates = []
    for analysis in analyses:
        for i, row in enumerate(analysis['forwards']):
            if row.get('status') != 'PAIRED': continue
            if stage == 'decode':
                accept = row['stage'] == 'DECODE' and row['fields'].get('bs') == 128 and row.get('decode_replay_proven')
            else:
                accept = (row['stage'] == 'EXTEND' and sum(row.get('graph_launch_calls_by_category', {}).values()) == 0
                          and sum(row.get('kernel_launch_calls_by_category', {}).values()) > 0)
            if accept: candidates.append((row['gpu_start_timestamp_us'], analysis['source']['path'], i, analysis, row))
    return min(candidates, key=lambda v: v[:3]) if candidates else None


def render_selected(selected, output, title, png):
    _, _, index, analysis, row = selected
    spec = importlib.util.spec_from_file_location('retained_trace_renderer', ROOT.parent / 'rubin-gb300-qwen3/render_trace.py')
    renderer = importlib.util.module_from_spec(spec); spec.loader.exec_module(renderer)
    trace = Path(analysis['source']['path'])
    with gzip.open(trace, 'rt') as stream: doc = json.load(stream, parse_float=Decimal)
    origin = min(e['ts'] for e in doc['traceEvents'] if analyzer.complete(e)
        and e.get('cat') in analyzer.GPU and e.get('args', {}).get('device', e.get('pid')) == row['device'])
    start_ms = (row['gpu_start_timestamp_us'] - origin) / 1000
    result = renderer.render(trace, output, title, float(start_ms), float(row['gpu_duration_ms']),
                             device=str(row['device']), max_visible_events=20000)
    if png:
        subprocess.run([str(renderer.NODE), str(renderer.ROOT / 'capture_trace.mjs'),
                        str(output / 'index.html'), str(output / 'trace.png')], check=True, timeout=90)
        result['png'] = str(output / 'trace.png')
    result.update(selection_rule='Earliest complete paired ON EXTEND with eager launch evidence, or ON DECODE bs=128 with replay proof; no duration ranking.',
                  analyzer_forward_index=index, selected_forward=row, start_ms_relative_to_first_device_gpu_event=start_ms)
    write(output / 'selection.json', result)
    return result


def build(experiment, inputs, output, png=False):
    experiment = load(experiment); entries = load(inputs)
    if output.exists(): raise ValueError('Fresh output directory required; canonical inputs are never overwritten')
    bindings = {b['label']: b for b in experiment['run_bindings']}
    if len({e['platform'] for e in entries}) != len(entries): raise ValueError('Duplicate platform input')
    prepared = []
    for entry in entries:
        label = entry['platform']
        if label not in bindings: raise ValueError('Unbound platform')
        store = Retained(Path(entry['directory']))
        ident, plan, refs = identity(store, bindings[label], entry['capture_run_id'])
        frozen = store.read('diagnostic/frozen-token-input.json') if 'diagnostic/frozen-token-input.json' in store.files else None
        if frozen and plan['host_train_sha256'] and frozen['source']['dataset_sha256'] != plan['host_train_sha256']:
            raise ValueError('Diagnostic dataset differs from the bound main')
        modes, analyses = {}, {}
        for mode in ['off', 'on']: modes[mode], analyses[mode] = condition(store, mode, ident, frozen)
        if all(modes.values()):
            for key in ['image_digest', 'initial_model_id', 'request_sha256', 'context_sha256', 'cache_policy', 'prefill_graph']:
                if modes['off'][key] != modes['on'][key]: raise ValueError('OFF/ON mismatch: ' + key)
        terminal = store.read('diagnostic/terminal.json') if 'diagnostic/terminal.json' in store.files else {'status': 'MISSING'}
        prepared.append((label, store, ident, plan, refs, modes, analyses, terminal))
    output.mkdir(parents=True)
    profiles = {'schema': 'qwen3-cudagraph-profiles-v1', 'experiment_id': experiment['experiment_id'],
                'status': 'pending', 'graph_evidence': [], 'profiles': []}
    diagnostics = {'schema': 'qwen3-cudagraph-diagnostics-v1', 'experiment_id': experiment['experiment_id'], 'pairs': []}
    for label, store, ident, plan, refs, modes, analyses, terminal in prepared:
        receipt_path = output / label / 'evidence.json'
        receipt = {'schema': 'retained-diagnostic-report-evidence-v1', 'platform': label,
            'source_run_id': bindings[label]['run_id'], 'capture_run_id': ident['run_id'], 'identity': ident,
            'retained_directory': str(store.root), 'retention': refs, 'checked_node_artifacts': store.checked, 'wrapper_terminal': terminal,
            'raw_modes': modes, 'analyses': analyses, 'wrapper_order': plan.get('order'),
            'utility_sha256': sha(Path(__file__)), 'analyzer_sha256': sha(LAB / 'analyze_sglang_graph_trace.py'),
            'limitations': ['Single TP1 initial-policy diagnostic, not four-engine main throughput.',
                'HTTP timings include prefill/decode/queue/response; profiled forward durations include profiler overhead.',
                'Initial input equality does not prove equal per-forward contexts, generated tokens or expert routing.',
                'No graph-launch evidence means unknown, never an inferred eager fallback.']}
        write(receipt_path, receipt); refs = [ref(receipt_path), *refs]
        bound = {'source_run_id': bindings[label]['run_id'], 'capture_run_id': ident['run_id']}
        off_decode = [r for a in analyses['off'] for r in a['forwards'] if r['stage'] == 'DECODE']
        off_eager = bool(off_decode) and all(r.get('status') == 'PAIRED'
            and not sum(r.get('graph_launch_calls_by_category', {}).values())
            and sum(r.get('kernel_launch_calls_by_category', {}).values()) > 0 for r in off_decode)
        on_replay = any(r.get('decode_replay_proven') for a in analyses['on'] for r in a['forwards'])
        pair_verified = all(modes.values()) and off_eager and on_replay
        diagnostics['pairs'].append({'platform': label, **bound, 'verified': pair_verified,
            'actual_trace_condition_proof': {'off_eager_decode_observed': off_eager, 'on_decode_replay_observed': on_replay},
            'status': 'available' if pair_verified else 'pending', 'wrapper_terminal_status': terminal['status'], 'scope': 'One TP1 engine; 128 frozen requests × 64 output tokens; three cold-radix HTTP batches per mode, including prefill/decode/queue/response.',
            'evidence_refs': refs, **modes})
        rows = [r for a in analyses['on'] for r in a['forwards']]
        decode = [r for r in rows if r['stage'] == 'DECODE']; replay = sum(bool(r.get('decode_replay_proven')) for r in decode)
        prefill = [r for r in rows if r['stage'] == 'EXTEND']
        profiles['graph_evidence'].append({'run_label': label, **bound, 'verified': bool(rows),
            'capture_status': 'actual_on_trace' if rows else 'pending', 'scope': 'Only retained ON diagnostic GPU forward annotations; neither main-run totals nor a fallback rate.',
            'decode_forwards': len(decode), 'decode_graph_replays': replay, 'decode_fallbacks': 0,
            'decode_unknown': len(decode)-replay, 'prefill_forwards': len(prefill),
            'prefill_graph_replays': sum(bool(sum(r.get('graph_launch_calls_by_category', {}).values())) for r in prefill),
            'evidence_refs': refs})
        for stage in ['prefill', 'decode']:
            record = {'run_label': label, **bound, 'stage': stage, 'verified': False, 'decode_graph': True,
                      'prefill_graph': False, 'title': f'{label.upper()} ON diagnostic: {stage}',
                      'scope': 'Pending complete paired ON ' + ('DECODE bs=128 replay proof.' if stage == 'decode' else 'EXTEND eager launch evidence.')}
            chosen = select_forward(analyses['on'], stage)
            if chosen:
                rendered = render_selected(chosen, output / label / stage, record['title'], png)
                row = chosen[-1]; source = chosen[-2]['source']
                record.update(verified=True, trace=source['path'], source_trace_sha256=source['sha256'],
                    scope=f"ON diagnostic; earliest qualifying {row['annotation']}; one selected forward. Full mixed/stage trace retained.",
                    observations=[f"GPU annotation: {float(row['gpu_duration_ms']):.3f} ms.",
                        f"Recorded kernel union: {float(row['kernel_intervals']['union_ms']):.3f} ms; not utilization.",
                        'Actual graph replay observed.' if stage == 'decode' else 'Kernel launches observed; no graph launch in this paired EXTEND.'],
                    caption='Rendered from actual retained trace. Instrumented one-engine scope; context/routing not proven matched across platforms.',
                    selection=rendered, evidence_refs=refs)
                if rendered.get('png'): record['image'] = rendered['png']
            profiles['profiles'].append(record)
    expected = len(bindings)*2
    verified_count = sum(bool(p['verified']) for p in profiles['profiles'])
    profiles['status'] = 'available' if verified_count == expected and expected else 'partial' if verified_count else 'pending'
    write(output / 'profiles.json', profiles); write(output / 'diagnostics.json', diagnostics)
    return {'output': str(output), 'profile_status': profiles['status'], 'verified_profiles': sum(p['verified'] for p in profiles['profiles']),
            'verified_pairs': sum(p['verified'] for p in diagnostics['pairs'])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, default=ROOT / 'experiment.json')
    p.add_argument('--inputs', type=Path, required=True, help='JSON list: platform, directory (retained operator root), capture_run_id')
    p.add_argument('--output', type=Path, required=True, help='New directory only')
    p.add_argument('--png', action='store_true', help='Render actual selected traces with local Chrome, timeout90s/image')
    a = p.parse_args(); print(json.dumps(build(a.experiment, a.inputs, a.output.resolve(), a.png), indent=2))


if __name__ == '__main__': main()
