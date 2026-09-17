#!/usr/bin/env python3
"""Slide-sized rendering of the full renderer's original selected GPU events."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise ValueError(message)


def close(a, b, tolerance=0.0001):
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=tolerance)


def prepare(summary, row):
    """Check the native chart values against the independently parsed forward."""
    events = summary['timeline']
    require(summary['status'] == 'captured' and not summary['visible_event_cap_applied']
            and len(events) == summary['counts']['selected_gpu_events']
            == row['all_gpu_intervals']['events_intersecting'], 'Truncated or mismatched GPU event window')
    require(close(summary['window']['duration_ms'], row['gpu_duration_ms'])
            and close(summary['window']['start_timestamp_us'], row['gpu_start_timestamp_us'], 0.002),
            'Selected forward/window differs')
    grouped = defaultdict(lambda: [0, 0.0])
    spans = []
    for event in events:
        if event['kind'] == 'gpu_kernel':
            grouped[event['name']][0] += 1
            grouped[event['name']][1] += event['clipped_duration_ms']
            spans.append((event['relative_start_ms'], event['relative_start_ms'] + event['clipped_duration_ms']))
    require(len(grouped) == len(summary['top_kernels']), 'Missing or duplicate kernel totals')
    for kernel in summary['top_kernels']:
        calls, duration = grouped[kernel['name']]
        require(calls == kernel['calls_intersecting_window'] and close(duration, kernel['clipped_duration_ms'], 1e-8),
                'Kernel bar differs from original events')
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    cumulative = sum(v[1] for v in grouped.values())
    union = sum(end - start for start, end in merged)
    require(len(spans) == row['kernel_intervals']['events_intersecting']
            and close(cumulative, row['kernel_intervals']['cumulative_ms'])
            and close(union, row['kernel_intervals']['union_ms']), 'Kernel counts/sum/union differ from forward analysis')
    # Numeric row IDs keep distinct kernels distinct even when displayed prefixes collide.
    top = [dict(k, row_id=i, prefix=k['name'][:41] + ('…' if len(k['name']) > 41 else ''))
           for i, k in enumerate(summary['top_kernels'][:6])]
    return {'schema': 'trtllm-compact-profile-slide-v1', 'width': 1440, 'height': 520,
            'source': summary['source'], 'window': summary['window'], 'timeline': events,
            'tracks': summary['tracks'], 'counts': summary['counts'], 'top_kernels': top,
            'kernel_sum_ms': cumulative, 'kernel_union_ms': union,
            'alignment': {'status': 'PASS', 'all_selected_gpu_events_preserved': True,
                          'kernel_bars_verified_against_events': True,
                          'forward_kernel_counts_sum_union_verified': True,
                          'float_renderer_tolerance_ms': 0.0001},
            'semantics': 'Same selected forward and all native GPU events; kernel bars sum clipped event durations, '
                         'not elapsed time or utilization. Full kernel names are retained in hover and JSON.'}


def render(directory, row, png, node, capture_script):
    summary_path = directory / 'trace-summary.json'
    data = prepare(json.loads(summary_path.read_text()), row)
    data['native_summary_sha256'] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    data['renderer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (directory / 'compact-slide-data.json').write_text(json.dumps(data, indent=2) + '\n')
    payload = json.dumps(data, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    shutil.copyfile(HERE / 'compact_slide.js', directory / 'compact_slide.js')
    html = (HERE / 'compact_slide.html').read_text().replace('__TRACE_DATA__', payload)
    (directory / 'compact-slide.html').write_text(html)
    result = {'compact_html': str(directory / 'compact-slide.html'),
              'compact_data': str(directory / 'compact-slide-data.json'), 'compact_alignment': data['alignment']}
    if png:
        target = directory / 'compact-slide.png'
        subprocess.run([str(node), str(capture_script), str(directory / 'compact-slide.html'), str(target)],
                       check=True, timeout=90)
        result.update(png=str(target), image_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    return result
