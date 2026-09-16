#!/usr/bin/env python3
"""Offline per-forward CUDA Graph trace analysis; no runtime operations.

Use an exact retained compressed-file SHA. Forward windows require matching CPU
and GPU user annotations; missing matches remain unknown. Interval unions are
observed trace coverage, never physical GPU utilization or occupancy.
"""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

STEP = re.compile(r'^step\[([A-Z_]+)\b([^\]]*)\]')
FIELD = re.compile(r'\b([a-z_]+)=([0-9]+)\b')
GPU = {'kernel', 'gpu_memcpy', 'gpu_memset'}
GRAPH = re.compile(r'^(?:cuda|cu)GraphLaunch(?:_ptsz|_v[0-9]+)?$')
MAX_BYTES = 128 * 1024**2


def finite(x):
    return type(x) in (int, float, Decimal) and math.isfinite(x)


def complete(e):
    return (isinstance(e, dict) and e.get('ph') == 'X' and finite(e.get('ts'))
            and finite(e.get('dur')) and e['dur'] > 0)


def intervals(events, begin, end):
    spans = sorted((max(e['ts'], begin), min(e['ts'] + e['dur'], end))
                   for e in events if e['ts'] < end and e['ts'] + e['dur'] > begin)
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    union = sum(b-a for a, b in merged)
    cumulative = sum(b-a for a, b in spans)
    return {'events_intersecting': len(spans), 'window_ms': (end-begin)/1000,
            'cumulative_ms': cumulative/1000, 'union_ms': union/1000,
            'uncovered_ms': (end-begin-union)/1000,
            'overlap_ms': (cumulative-union)/1000}


def group(name):
    if name == 'fused_moe_kernel': return 'fused_moe_compute'
    if name in {'_fwd_grouped_kernel_stage1', '_fwd_kernel_stage2'}: return 'triton_decode_attention'
    if any(s in name for s in ['moe_align_block_size_kernel', 'count_and_sort_expert_tokens_kernel',
                               'moe_sum_reduce_kernel', 'sglang::act_and_mul_kernel', '_router_triton_kernel']):
        return 'moe_routing_activation_reduction'
    if 'cublas' in name.lower(): return 'explicit_cublas_name'
    if name.startswith('nvjet_'): return 'nvjet_name_no_cpu_mm_attribution'
    if 'flashinfernormkernels' in name: return 'flashinfer_norm_name'
    if 'sglang::fused_qknorm' in name or 'sglang::fused_rope_store' in name: return 'qknorm_rope'
    return 'other_kernel'


def analyze_document(doc):
    all_events = doc.get('traceEvents')
    if not isinstance(all_events, list): raise ValueError('Expected Chrome traceEvents')
    events = [e for e in all_events if complete(e)]
    cpu = [e for e in events if e.get('cat') == 'user_annotation' and STEP.match(e.get('name', ''))]
    gpu = sorted((e for e in events if e.get('cat') == 'gpu_user_annotation' and STEP.match(e.get('name', ''))), key=lambda e:e['ts'])
    rows = []
    for g in gpu:
        fields = {k:int(v) for k,v in FIELD.findall(STEP.match(g['name'])[2])}
        row = {'annotation': g['name'], 'stage': STEP.match(g['name'])[1], 'fields': fields,
               'external_id': g.get('args', {}).get('External id'), 'device': g.get('args', {}).get('device', g.get('pid')),
               'gpu_start_timestamp_us': g['ts'], 'gpu_duration_ms': g['dur']/1000}
        matches = [c for c in cpu if row['external_id'] is not None and
                   c.get('args', {}).get('External id') == row['external_id'] and c['name'] == g['name']]
        if len(matches) != 1:
            row.update(status='UNKNOWN_CPU_GPU_PAIR', cpu_matches=len(matches));rows.append(row);continue
        c = matches[0]
        begin, end = g['ts'], g['ts'] + g['dur']
        device_events = [e for e in events if e.get('cat') in GPU and
                         e.get('args', {}).get('device', e.get('pid')) == row['device']]
        kernels = [e for e in device_events if e['cat'] == 'kernel']
        selected = [e for e in kernels if e['ts'] < end and e['ts'] + e['dur'] > begin]
        groups, names = defaultdict(list), defaultdict(list)
        for e in selected:
            groups[group(e['name'])].append(e);names[e['name']].append(e)
        api = [e for e in events if e.get('cat') in {'cuda_runtime', 'cuda_driver'} and
               e.get('pid') == c.get('pid') and e.get('tid') == c.get('tid') and
               c['ts'] <= e['ts'] and e['ts'] + e['dur'] <= c['ts'] + c['dur']]
        launches = [e for e in api if GRAPH.fullmatch(e.get('name', ''))]
        row.update(status='PAIRED' if selected else 'PAIRED_NO_KERNELS', cpu_duration_ms=c['dur']/1000,
                   kernel_intervals=intervals(kernels, begin, end), all_gpu_intervals=intervals(device_events, begin, end),
                   kernel_groups={k:intervals(v, begin, end) for k,v in sorted(groups.items())},
                   top_kernels=sorted([{'name':k, **intervals(v, begin, end)} for k,v in names.items()], key=lambda x:x['cumulative_ms'], reverse=True)[:20],
                   cpu_api_counts=dict(Counter((e.get('cat', '') + ':' + e['name']) for e in api)),
                   graph_launch_calls_by_category=dict(Counter(e['cat'] for e in launches)),
                   kernel_launch_calls_by_category=dict(Counter(e['cat'] for e in api if 'LaunchKernel' in e.get('name', ''))),
                   decode_replay_proven=row['stage']=='DECODE' and bool(launches) and bool(selected))
        rows.append(row)
    return {'schema':'sglang-cudagraph-forward-analysis-v1', 'raw_event_count':len(all_events),
            'complete_categories':dict(Counter(e.get('cat', '') for e in events)),
            'cpu_forward_annotations':len(cpu), 'gpu_forward_annotations':len(gpu), 'forwards':rows,
            'device_properties':doc.get('deviceProperties', []),
            'limitations':[
                'Each duration includes profiling overhead. CPU and GPU annotation spans are different measurements.',
                'Kernel cumulative sums can overlap. Union coverage and uncovered time do not measure physical GPU idle, occupancy or utilization.',
                'CUDA Graph proof requires a graph launch inside the matched same-thread CPU DECODE annotation and GPU kernels in its matched GPU annotation.',
                'Detailed fields retain observed batch/token aggregates; aggregate context fields alone do not prove identical per-request KV lengths or expert routing.',
                'GPU kernels are selected by device and temporal overlap, not attributed exclusively to an engine when another workload shares the device.',
                'Runtime and driver launch API counts stay separate because nested calls can describe the same launch.',
                'Graph capture can remove CPU-to-kernel operation links; name-based groups are explicitly named and do not invent aten::mm attribution.',
                'Missing CPU/GPU annotation pairs stay unknown and must not be presented as zero-duration forwards.',
            ]}


def analyze(path, expected_sha):
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha: raise ValueError('Exact retained trace SHA mismatch')
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as stream: decoded = stream.read(MAX_BYTES+1)
    else: decoded = raw
    if len(decoded)>MAX_BYTES: raise ValueError('Decoded trace exceeds128MiB bound')
    doc=json.loads(decoded, parse_float=Decimal)
    result=analyze_document(doc)
    result['source']={'path':str(path.resolve()),'sha256':actual,'compressed_bytes':len(raw),
                      'analyzer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('trace',type=Path)
    p.add_argument('--sha256',required=True);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();result=analyze(a.trace,a.sha256)
    a.output.write_text(json.dumps(result,indent=2,default=float,allow_nan=False)+'\n')
    print(json.dumps({'output':str(a.output),'forwards':len(result['forwards']),
                      'proven_decode_replays':sum(x.get('decode_replay_proven',False) for x in result['forwards'])}))


if __name__=='__main__':main()
