#!/usr/bin/env python3
"""Render real Torch/Chrome complete events as an offline timeline and optional PNG.

Timestamps and durations use Chrome Trace Event microseconds. Window arguments are
milliseconds relative to the first GPU event (default) or first complete event.
Only complete events (ph=X) contribute. CUDA runtime launches remain CPU events.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
NODE = Path('/Users/kaixih/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
GPU_CATEGORIES = {'kernel': 'gpu_kernel', 'gpu_memcpy': 'gpu_memcpy', 'gpu_memset': 'gpu_memset'}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def classify(event):
    categories = {part.strip().lower() for part in str(event.get('cat', '')).split(',')}
    for category, kind in GPU_CATEGORIES.items():
        if category in categories:
            return kind
    if categories & {'cuda_runtime', 'cuda_driver', 'cpu_op', 'python_function', 'user_annotation', 'ac2g'}:
        return 'cpu'
    if 'gpu_user_annotation' in categories:
        return 'gpu_annotation'
    return 'unclassified'


def summarize_trace(document, start_ms=0.0, duration_ms=50.0, origin='gpu', device=None, max_visible_events=4000):
    if not finite(start_ms) or not finite(duration_ms) or duration_ms <= 0 or max_visible_events < 1:
        raise ValueError('start_ms must be finite; duration_ms and max_visible_events must be positive')
    events = document.get('traceEvents', []) if isinstance(document, dict) else document
    if not isinstance(events, list):
        raise ValueError('Trace must be an event array or an object with traceEvents')
    complete, kinds, invalid = [], Counter(), 0
    for event in events:
        if not isinstance(event, dict) or event.get('ph') != 'X':
            continue
        if not finite(event.get('ts')) or not finite(event.get('dur')) or event['dur'] < 0:
            invalid += 1
            continue
        kind = classify(event)
        kinds[kind] += 1
        args = event.get('args') if isinstance(event.get('args'), dict) else {}
        complete.append({'name': str(event.get('name', '(unnamed)')), 'category': str(event.get('cat', '')),
                         'kind': kind, 'ts': event['ts'], 'dur': event['dur'],
                         'device': args.get('device', event.get('pid', 'unknown')),
                         'stream': args.get('stream', event.get('tid', 'unknown')),
                         'pid': event.get('pid'), 'tid': event.get('tid')})
    gpu = [event for event in complete if event['kind'] in GPU_CATEGORIES.values()
           and (device is None or str(event['device']) == str(device))]
    origin_events = gpu if origin == 'gpu' and gpu else complete
    origin_us = min((event['ts'] for event in origin_events), default=0)
    begin, end = origin_us + start_ms * 1000, origin_us + (start_ms + duration_ms) * 1000
    window = [event for event in complete if event['ts'] < end and event['ts'] + event['dur'] > begin]
    window_counts = Counter(event['kind'] for event in window)
    selected = [event for event in gpu if event['ts'] < end and event['ts'] + event['dur'] > begin]
    selected.sort(key=lambda event: (event['ts'], str(event['device']), str(event['stream'])))
    totals = defaultdict(lambda: {'calls_intersecting_window': 0, 'clipped_duration_us': 0.0, 'full_duration_us': 0.0})
    for event in selected:
        clipped = min(event['ts'] + event['dur'], end) - max(event['ts'], begin)
        event['relative_start_ms'] = (max(event['ts'], begin) - origin_us) / 1000
        event['clipped_duration_ms'] = clipped / 1000
        event['full_duration_ms'] = event['dur'] / 1000
        event['track'] = f"GPU {event['device']} / stream {event['stream']}"
        if event['kind'] == 'gpu_kernel':
            totals[event['name']]['calls_intersecting_window'] += 1
            totals[event['name']]['clipped_duration_us'] += clipped
            totals[event['name']]['full_duration_us'] += event['dur']
    top = [{'name': name, **value, 'clipped_duration_ms': value['clipped_duration_us'] / 1000}
           for name, value in totals.items()]
    top.sort(key=lambda item: item['clipped_duration_us'], reverse=True)
    streams = sorted({event['track'] for event in selected})
    return {'schema': 'torch-chrome-trace-render-v1',
            'status': 'captured' if selected else 'no_gpu_events_in_window',
            'timestamp_unit': 'microseconds', 'origin_requested': origin,
            'origin_used': 'first_selected_gpu_event' if origin == 'gpu' and gpu else 'first_complete_event',
            'origin_timestamp_us': origin_us,
            'window': {'start_ms_relative_to_origin': start_ms, 'duration_ms': duration_ms,
                       'start_timestamp_us': begin, 'end_timestamp_us': end, 'device_filter': device},
            'counts': {'raw_events': len(events), 'valid_complete_events': len(complete),
                       'invalid_complete_events': invalid, 'complete_events_by_kind': dict(kinds),
                       'window_complete_events_by_kind_all_devices': dict(window_counts),
                       'selected_gpu_events': len(selected),
                       'visible_gpu_events': min(len(selected), max_visible_events)},
            'visible_event_cap_applied': len(selected) > max_visible_events,
            'tracks': streams, 'timeline': selected[:max_visible_events], 'top_kernels': top,
            'semantics': [
                'Rendered from original trace events. This is not an Nsight UI screenshot.',
                'Only ph=X complete events contribute. B/E, counters, flow arrows and annotations are not kernel durations.',
                'GPU classification requires cat=kernel, gpu_memcpy or gpu_memset. CUDA runtime/driver events remain CPU.',
                'Kernel totals use duration clipped to this window. Calls count kernels that intersect the window.',
                'Cumulative kernel durations may overlap across streams and exceed elapsed time. They are not wall utilization.',
                'Unknown event categories remain unclassified. A missing category is never assumed to be a GPU kernel.',
                'The visible event cap affects the timeline only. Top-kernel totals use every selected GPU kernel.',
            ]}


def render(trace, output, title='GPU timeline from a Torch trace', start_ms=0.0, duration_ms=50.0,
           origin='gpu', device=None, max_visible_events=4000, png=False):
    trace, output = trace.resolve(), output.resolve()
    opener = gzip.open if trace.suffix.lower() == '.gz' else open
    with opener(trace, 'rt', encoding='utf-8') as stream:
        document = json.load(stream)
    summary = summarize_trace(document, start_ms, duration_ms, origin, device, max_visible_events)
    summary.update(title=title, source={'path': str(trace), 'sha256': file_hash(trace), 'bytes': trace.stat().st_size})
    output.mkdir(parents=True, exist_ok=True)
    for name in ['trace_view.html', 'trace_view.js', 'plotly-3.1.0.min.js']:
        shutil.copyfile(ROOT / 'assets' / name, output / name)
    payload = json.dumps(summary, ensure_ascii=True, allow_nan=False)
    (output / 'trace-summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=True, allow_nan=False) + '\n')
    template = (output / 'trace_view.html').read_text()
    safe = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    (output / 'index.html').write_text(template.replace('__TRACE_DATA__', safe))
    (output / 'trace_view.html').unlink()  # Only our newly created template copy.
    if png:
        subprocess.run([str(NODE), str(ROOT / 'capture_trace.mjs'), str(output / 'index.html'),
                        str(output / 'trace.png')], check=True)
    return {'html': str(output / 'index.html'), 'summary': str(output / 'trace-summary.json'),
            'png': str(output / 'trace.png') if png else None, 'status': summary['status'],
            'selected_gpu_events': summary['counts']['selected_gpu_events'], 'source_sha256': summary['source']['sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--title', default='GPU timeline from a Torch trace')
    parser.add_argument('--start-ms', type=float, default=0.0)
    parser.add_argument('--duration-ms', type=float, default=50.0)
    parser.add_argument('--origin', choices=['gpu', 'trace'], default='gpu')
    parser.add_argument('--device', help='Select a device value recorded in trace args.device')
    parser.add_argument('--max-visible-events', type=int, default=4000)
    parser.add_argument('--png', action='store_true', help='Capture the offline HTML using local headless Chrome')
    args = parser.parse_args()
    print(json.dumps(render(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
