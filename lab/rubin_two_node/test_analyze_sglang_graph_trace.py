import unittest
import analyze_sglang_graph_trace as a


def event(cat,name,ts,dur,pid,tid,external=None):
    return {'cat':cat,'name':name,'ts':ts,'dur':dur,'pid':pid,'tid':tid,'ph':'X','args':{'External id':external}}


def fixture():
    return [event('user_annotation','step[DECODE bs=128 g_sq=128]',0,20,9,3,5),
            event('gpu_user_annotation','step[DECODE bs=128 g_sq=128]',5,10,0,7,5),
            event('kernel','fused_moe_kernel',6,4,0,7),event('kernel','other',8,5,0,8),
            event('cuda_runtime','cudaGraphLaunch',3,1,9,3)]


class TraceTests(unittest.TestCase):
    def test_overlap_is_not_additive_wall_coverage(self):
        x=a.analyze_document({'traceEvents':fixture()})['forwards'][0]
        self.assertTrue(x['decode_replay_proven'])
        self.assertEqual(x['kernel_intervals']['cumulative_ms'],.009)
        self.assertEqual(x['kernel_intervals']['union_ms'],.007)
        self.assertEqual(x['kernel_intervals']['uncovered_ms'],.003)
        self.assertEqual(x['fields'],{'bs':128,'g_sq':128})

    def test_other_thread_or_partial_api_is_not_replay_proof(self):
        for change in [{'tid':4},{'ts':19,'dur':2}]:
            e=fixture();e[-1].update(change)
            self.assertFalse(a.analyze_document({'traceEvents':e})['forwards'][0]['decode_replay_proven'])

    def test_missing_annotation_pair_is_unknown(self):
        e=fixture();e[0]['args']['External id']=6
        x=a.analyze_document({'traceEvents':e})['forwards'][0]
        self.assertEqual(x['status'],'UNKNOWN_CPU_GPU_PAIR')
        self.assertNotIn('kernel_intervals',x)

    def test_nested_api_categories_stay_separate(self):
        e=fixture()+[event('cuda_driver','cuGraphLaunch',3.2,.5,9,3)]
        x=a.analyze_document({'traceEvents':e})['forwards'][0]
        self.assertEqual(x['graph_launch_calls_by_category'],{'cuda_runtime':1,'cuda_driver':1})


if __name__=='__main__':unittest.main()
