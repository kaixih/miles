"""CPU-only source/value checks for the slide rendering of real trace events."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_slide


def fixture():
    names = ['bmm_' + 'same_long_prefix_' * 8 + suffix for suffix in ('first', 'second')]
    events = [{'name': name, 'kind': 'gpu_kernel', 'clipped_duration_ms': duration,
               'relative_start_ms': start, 'track': 'GPU 0 / stream 13'}
              for name, duration, start in zip(names, (2.0, 1.0), (0.0, 3.0))]
    summary = {'status': 'captured', 'visible_event_cap_applied': False, 'timeline': events,
               'counts': {'selected_gpu_events': 2}, 'tracks': ['GPU 0 / stream 13'], 'source': {},
               'window': {'duration_ms': 10.0, 'start_timestamp_us': 0.0},
               'top_kernels': [{'name': e['name'], 'clipped_duration_ms': e['clipped_duration_ms'],
                                'calls_intersecting_window': 1} for e in events]}
    row = {'gpu_duration_ms': 10, 'gpu_start_timestamp_us': 0,
           'all_gpu_intervals': {'events_intersecting': 2},
           'kernel_intervals': {'events_intersecting': 2, 'cumulative_ms': 3, 'union_ms': 3}}
    return summary, row


class Slide(unittest.TestCase):
    def test_colliding_prefixes_keep_distinct_rows_and_original_totals(self):
        summary, row = fixture()
        original = copy.deepcopy(summary)
        result = render_slide.prepare(summary, row)
        self.assertEqual(summary, original)
        self.assertEqual(result['top_kernels'][0]['prefix'], result['top_kernels'][1]['prefix'])
        self.assertEqual([k['row_id'] for k in result['top_kernels']], [0, 1])
        self.assertEqual([k['clipped_duration_ms'] for k in result['top_kernels']], [2, 1])
        self.assertNotEqual(result['top_kernels'][0]['name'], result['top_kernels'][1]['name'])
        self.assertEqual(result['timeline'], original['timeline'])

    def test_bar_value_must_come_from_events(self):
        summary, row = fixture()
        summary['top_kernels'][0]['clipped_duration_ms'] = 3
        with self.assertRaisesRegex(ValueError, 'bar differs'):
            render_slide.prepare(summary, row)

    def test_capped_timeline_is_rejected(self):
        summary, row = fixture()
        summary['visible_event_cap_applied'] = True
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            render_slide.prepare(summary, row)

    def test_union_is_checked_independently(self):
        summary, row = fixture()
        row['kernel_intervals']['union_ms'] = 2
        with self.assertRaisesRegex(ValueError, 'counts/sum/union'):
            render_slide.prepare(summary, row)


if __name__ == '__main__':
    unittest.main()
