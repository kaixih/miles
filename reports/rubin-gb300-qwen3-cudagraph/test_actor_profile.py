"""Temporary synthetic fixtures only; no runtime or canonical artifact access."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('actor_deck',ROOT/'generate.py')
G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)


def fixture(root):
    experiment=json.loads((ROOT/'experiment.json').read_text())
    experiment['experiment_id']='SYNTHETIC_ACTOR_QA_ONLY'
    experiment['run_bindings']=[{'label':p,'run_id':'SYNTHETIC_MAIN_'+p} for p in ['rubin','gb300']]
    def write(name,value):
        path=root/name;path.write_text(json.dumps(value));return path
    receipt=write('SYNTHETIC_receipt.json',{'purpose':'CPU schema/layout fixture; never measured data'})
    runs=[]
    for i,platform in enumerate(['rubin','gb300']):
        trace=write('SYNTHETIC_'+platform+'.json',{'traceEvents':[],'purpose':'Synthetic fixture only'})
        runs.append({'run_label':platform,'source_run_id':'SYNTHETIC_MAIN_'+platform,
            'capture_run_id':'SYNTHETIC_CAPTURE_'+platform,'verified':True,
            'scope':'SYNTHETIC QA ONLY; rank0, two updates; no measurement',
            'rank_ids':[0],'input_identity':dict(zip(G.ACTOR_IDENTITIES,['a'*64,'b'*64,'c'*64])),
            'samples':[{'update_id':u,'durations_ms':{k:(i+1)*10+u for k in G.ACTOR_CATEGORIES}} for u in [0,1]],
            'trace':str(trace),'source_trace_sha256':G.sha256(trace),
            'evidence_refs':[{'path':str(receipt),'sha256':G.sha256(receipt)}]})
    actor={'schema':'qwen3-actor-profile-v1','experiment_id':experiment['experiment_id'],
           'measurement_basis':'gpu_kernel_interval_union_ms','capture_window':'whole_optimizer_update','runs':runs,
           'scope':'SYNTHETIC QA ONLY: rank0, two updates. Not measured evidence.',
           'matched_workload':{'verified':False}}
    return experiment,actor


class ActorEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='SYNTHETIC_ACTOR_TEST_');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.experiment,self.actor=fixture(self.root)

    def write(self,name,data):
        path=self.root/name;path.write_text(json.dumps(data));return path

    def build(self):
        G.build_report(experiment=self.write('experiment.json',self.experiment),
            actor_profile=self.write('actor.json',self.actor),output=self.root/'site')
        return json.loads((self.root/'site/report-data.json').read_text())

    def match(self):
        self.actor['matched_workload']={'verified':True,
            'identity_scope':'Same initial construction and frozen batch; post-warmup tensors not bitwise verified.',
            'evidence_refs':copy.deepcopy(self.actor['runs'][0]['evidence_refs'])}

    def test_single_verified_unmatched_capture_survives_without_main_timing(self):
        self.actor['runs'].pop();out=self.build();actor=out['derived']['actor_profile']
        self.assertEqual(actor['status'],'available');self.assertEqual(actor['matching_status'],'diagnostic_unmatched')
        self.assertEqual(actor['runs'][0]['mean_ms']['forward'],10.5)
        self.assertEqual(out['derived']['paired_timing']['count'],0)
        self.assertEqual(out['inputs']['actor_profile']['runs'][0]['attachments']['trace']['status'],'available')

    def test_differing_inputs_allow_individual_evidence_but_reject_matched_claim(self):
        self.actor['runs'][1]['input_identity']['batch_sha256']='d'*64
        self.assertEqual(self.build()['derived']['actor_profile']['matching_status'],'diagnostic_unmatched')
        self.match()
        with self.assertRaisesRegex(ValueError,'batch_sha256'):self.build()

    def test_matched_workload_does_not_require_post_warmup_bitwise_state(self):
        self.match();out=self.build()
        self.assertEqual(out['derived']['actor_profile']['matching_status'],'matched_workload')
        self.assertIn('not bitwise',out['derived']['actor_profile']['identity_scope'])
        self.assertEqual(out['derived']['actor_profile']['runs'][1]['mean_ms']['optimizer'],20.5)

    def test_unknown_category_is_not_zero_or_shortened_cohort(self):
        self.actor['runs'][0]['samples'][1]['durations_ms']['recompute']=None
        with self.assertRaisesRegex(ValueError,'attribution reason'):self.build()
        self.actor['runs'][0]['unattributed_reasons']={'recompute':'Range absent in one selected update'}
        self.assertIsNone(self.build()['derived']['actor_profile']['runs'][0]['mean_ms']['recompute'])

    def test_trace_receipt_and_image_hash_guards(self):
        for kind in ['trace','receipt','image']:
            with self.subTest(kind=kind):
                self.experiment,self.actor=fixture(self.root)
                row=self.actor['runs'][0]
                if kind=='trace':row['source_trace_sha256']='0'*64
                elif kind=='receipt':row['evidence_refs'][0]['sha256']='0'*64
                else:
                    image=self.root/'SYNTHETIC.png';image.write_bytes(b'fixture-not-an-image')
                    row.update(image=str(image),source_image_sha256='0'*64)
                with self.assertRaisesRegex(ValueError,'hash mismatch|SHA256 mismatched'):self.build()

    def microbatch(self):
        self.actor['capture_window']='single_forward_backward_microbatch'
        for row in self.actor['runs']:
            row['samples']=row['samples'][1:]
            row['samples'][0]['durations_ms']['optimizer']=None
            row['unattributed_reasons']={'optimizer':'Outside selected F/B microbatch'}
            row['microbatch_index']=1;row['final_gradient_sync']=None
            row['input_identity']['selected_microbatch_sha256']='d'*64

    def test_microbatch_uses_explicit_units_and_rejects_out_of_window_claims(self):
        self.microbatch();out=self.build()['derived']['actor_profile']
        self.assertIn('/ microbatch',out['measurement_label'])
        self.assertIsNone(out['runs'][0]['mean_ms']['optimizer'])
        for key,value in [('rank_ids',[1]),('microbatch_index',0),('final_gradient_sync',0)]:
            row=self.actor['runs'][0];old=row[key];row[key]=value
            with self.assertRaisesRegex(ValueError,'Microbatch trace'):self.build()
            row[key]=old
        self.actor['runs'][0]['samples'][0]['durations_ms']['optimizer']=1
        with self.assertRaisesRegex(ValueError,'Optimizer must be null'):self.build()

    def test_matched_microbatch_requires_selected_packing_identity(self):
        self.microbatch();self.match()
        self.assertEqual(self.build()['derived']['actor_profile']['matching_status'],'matched_workload')
        self.actor['runs'][1]['input_identity']['selected_microbatch_sha256']='e'*64
        with self.assertRaisesRegex(ValueError,'selected_microbatch_sha256'):self.build()
        self.actor['matched_workload']={'verified':False}
        self.assertEqual(self.build()['derived']['actor_profile']['matching_status'],'diagnostic_unmatched')

    def test_findings_require_verified_scope_and_audited_hashes(self):
        self.actor['findings']=[{'verified':True,'text':'SYNTHETIC QA ONLY: observed window, no causal attribution.',
            'run_labels':['rubin'],'evidence_refs':copy.deepcopy(self.actor['runs'][0]['evidence_refs'])}]
        self.actor['interpretation_limits']='SYNTHETIC QA ONLY: one rank; nested ranges overlap.'
        out=self.build()
        self.assertEqual(out['derived']['actor_profile']['findings'][0]['text'],self.actor['findings'][0]['text'])
        self.assertEqual(len(out['inputs']['actor_profile']['findings'][0]['verified_receipts']),1)
        self.actor['findings'][0]['evidence_refs'][0]['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'SHA256 mismatched'):self.build()

    def test_findings_reject_overflow_missing_audit_and_unverified_platform(self):
        base={'verified':True,'text':'SYNTHETIC observation','run_labels':['rubin'],
              'evidence_refs':copy.deepcopy(self.actor['runs'][0]['evidence_refs'])}
        for changed in [{'verified':False},{'text':'x'*151},{'text':'line1\nline2'},{'evidence_refs':[]},{'run_labels':['unknown']}]:
            self.actor['findings']=[{**base,**changed}]
            with self.assertRaisesRegex(ValueError,'Actor finding'):self.build()
        self.actor['findings']=[{**base,'text':'x'*110} for _ in range(2)]
        with self.assertRaisesRegex(ValueError,'combined text'):self.build()
        self.actor['findings']=[copy.deepcopy(base) for _ in range(3)]
        with self.assertRaisesRegex(ValueError,'at most two'):self.build()
        self.actor['findings']=[base];self.actor['runs'][0]['verified']=False
        with self.assertRaisesRegex(ValueError,'verified platform'):self.build()
        self.actor['findings']=[];self.actor['interpretation_limits']='x'*181
        with self.assertRaisesRegex(ValueError,'interpretation limits'):self.build()

    def test_bound_identity_and_update_scope_are_enforced(self):
        self.actor['runs'][0]['source_run_id']='OTHER_MAIN'
        with self.assertRaisesRegex(ValueError,'exact main run'):self.build()
        self.experiment,self.actor=fixture(self.root);self.match()
        self.actor['runs'][1]['samples'][0]['update_id']=2
        with self.assertRaisesRegex(ValueError,'rank/update scope'):self.build()

    def test_nonfinite_or_boolean_duration_is_rejected(self):
        self.actor['runs'][0]['samples'][0]['durations_ms']['forward']=True
        with self.assertRaisesRegex(ValueError,'finite nonnegative'):self.build()

    def test_omitted_input_keeps_original_provenance_sections(self):
        G.build_report(experiment=self.write('experiment.json',self.experiment),output=self.root/'site')
        out=json.loads((self.root/'site/report-data.json').read_text())
        self.assertNotIn('actor_profile',[x['section'] for x in out['provenance']])
        self.assertEqual(out['derived']['actor_profile']['status'],'pending')


if __name__=='__main__':unittest.main()
