"""Synthetic events test arithmetic only. No fixture enters the delivered slides."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('trace_renderer', ROOT/'render_trace.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def event(name, cat, ts, dur, stream=1, device=0):
    return {'ph':'X','name':name,'cat':cat,'ts':ts,'dur':dur,'pid':42,'tid':7,
            'args':{'device':device,'stream':stream}}


class TraceTests(unittest.TestCase):
    def test_overlap_and_clipping_are_duration_not_wall_utilization(self):
        data=module.summarize_trace({'traceEvents':[
            event('a','kernel',1000,1000),event('b','kernel',1500,1000,stream=2),
            event('cudaLaunchKernel','cuda_runtime',900,50),event('copy','gpu_memcpy',1900,50)]},
            start_ms=.25,duration_ms=1)
        totals={r['name']:r['clipped_duration_ms'] for r in data['top_kernels']}
        self.assertEqual(totals,{'a':.75,'b':.75})
        self.assertEqual(sum(totals.values()),1.5)
        self.assertEqual(data['counts']['complete_events_by_kind']['cpu'],1)
        self.assertEqual(data['counts']['selected_gpu_events'],3)
        self.assertEqual(data['timeline'][0]['relative_start_ms'],.25)

    def test_launch_names_alone_never_imply_gpu(self):
        self.assertEqual(module.classify(event('kernel','cuda_runtime',0,1)),'cpu')
        self.assertEqual(module.classify(event('gemm','',0,1)),'unclassified')
        self.assertEqual(module.classify(event('range','gpu_user_annotation',0,1)),'gpu_annotation')

    def test_device_filter_and_visible_cap_keep_full_aggregate(self):
        events=[event('same','kernel',i*1000,200,device=0) for i in range(3)]
        events += [event('other','kernel',0,200,device=1)]
        data=module.summarize_trace(events,duration_ms=4,device='0',max_visible_events=1)
        self.assertEqual(len(data['timeline']),1)
        self.assertTrue(data['visible_event_cap_applied'])
        self.assertEqual(data['top_kernels'][0]['calls_intersecting_window'],3)
        self.assertEqual(data['counts']['selected_gpu_events'],3)

    def test_invalid_and_incomplete_events_do_not_become_kernel_durations(self):
        data=module.summarize_trace([{'ph':'B','cat':'kernel','ts':0},event('bad','kernel',0,-1),
                                     event('bad2','kernel',float('nan'),1)])
        self.assertEqual(data['counts']['invalid_complete_events'],2)
        self.assertEqual(data['top_kernels'],[])
        self.assertEqual(data['status'],'no_gpu_events_in_window')
        with self.assertRaises(ValueError):module.summarize_trace([],duration_ms=0)


if __name__=='__main__':unittest.main()
