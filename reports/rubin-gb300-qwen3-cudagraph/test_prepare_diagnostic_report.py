"""Synthetic CPU fixtures only; no network, GPU or canonical report writes."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prepare_diagnostic_report', ROOT / 'prepare_diagnostic_report.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def event(cat, name, ts, dur, pid=1, tid=1, **args):
    return dict(ph='X', cat=cat, name=name, ts=ts, dur=dur, pid=pid, tid=tid, args=args)


def document(mode):
    events = []
    for index, (stage, bs, start, duration) in enumerate([('EXTEND', 128, 100, 80), ('DECODE', 128, 300, 80), ('DECODE', 128, 500, 30)]):
        name = f'step[{stage} bs={bs} c_sq=128 g_sk=4096]'
        events += [event('user_annotation', name, start, duration, **{'External id': index}),
                   event('gpu_user_annotation', name, start+1, duration-1, pid=0, device=0, **{'External id': index}),
                   event('cuda_runtime', 'cudaGraphLaunch' if mode == 'on' and stage == 'DECODE' else 'cudaLaunchKernel', start+2, 2),
                   event('kernel', 'SYNTHETIC_CPU_FIXTURE_KERNEL', start+6, 10, pid=0, device=0, stream=7)]
    return {'traceEvents': events, 'purpose': 'SYNTHETIC CPU fixture, never measured evidence'}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='SYNTHETIC_REPORT_ADAPTER_'); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name); self.root = self.base / 'retained'; self.node = self.root / 'node-output'
        self.node.mkdir(parents=True)
        self.capture = 'SYNTHETIC_rubin_capture'; self.main = 'SYNTHETIC_rubin_main'
        self.experiment = self.base / 'experiment.json'; self.inputs = self.base / 'inputs.json'
        self.write(self.experiment, {'schema': 'qwen3-cudagraph-experiment-v1', 'experiment_id': 'SYNTHETIC_ONLY',
            'run_bindings': [{'label': 'rubin', 'run_id': self.main}], 'previous_run_ids': []})
        self.write(self.inputs, [{'platform': 'rubin', 'directory': str(self.root), 'capture_run_id': self.capture}])
        image = 'fixture@sha256:a03106bdd90c5d6067fbff246fff25df979f9da8486eb0dac795a315a2346d6c'
        inv = {'config.json': {'bytes': 100, 'sha256': 'b'*64}, 'model.safetensors': {'bytes': 1000}}
        self.ident = {'run_id': self.capture, 'uid': 28644, 'gid': 30,
            'container_id': 'c'*64, 'container_name': 'miles-graph-diagnostic-'+self.capture,
            'image_reference': image, 'image_id': 'sha256:'+'d'*64,
            'labels': {'miles.graph_diagnostic': self.capture}, 'main_terminal_verified': True,
            'source_commits': {'miles_main': 'e'*40}, 'docker_inspect_verified_at': 'synthetic',
            'main_submission_id': 'SYNTHETIC_RAY_ID', 'initial_model_inventory': inv,
            'initial_model_id': M.capture.json_sha256(inv),
            'diagnostic_source_sha256': {'run_sglang_graph_diagnostic.py': 'f'*64, 'sglang_graph_capture.py': 'a'*64}}
        self.write(self.root/'recovery-plan.json', {'config': {'run_id': self.main, 'diagnostic_run': self.capture, 'image': image},
                   'main_submission_id': self.ident['main_submission_id']})
        self.write(self.node/'diagnostic-identity.json', self.ident)
        self.write(self.node/'diagnostic/plan.json', {'identity': self.ident, 'source_sha256': self.ident['diagnostic_source_sha256'], 'order': ['on','off']})
        self.frozen = {'input_ids': [[i+1, 5] for i in range(128)], 'source': {'dataset_sha256': 'a'*64,
            'selected_row_indices': list(range(128)), 'tokenizer_model': '/models/Qwen3-30B-A3B', 'chat_template_kwargs': {}}}
        self.write(self.node/'diagnostic/frozen-token-input.json', self.frozen)
        for mode in ['off', 'on']: self.make_mode(mode)
        self.write(self.node/'diagnostic/terminal.json', {'status': 'COMPLETED'})
        self.seal()

    def write(self, path, data): M.write(path, data)

    def make_mode(self, mode):
        folder = self.node/'diagnostic'/mode; folder.mkdir()
        engine = {'pid': 123, 'start_ticks': 321, 'run_id': self.capture}
        server = {'model_path': '/models/Qwen3-30B-A3B', 'tp_size': 1, 'pp_size': 1, 'base_gpu_id': 0,
                  'cuda_graph_config': {'decode': {'backend': 'full' if mode=='on' else 'disabled'}, 'prefill': {'backend': 'disabled'}}}
        self.write(folder/'server-info.json', {'status':200, 'body': json.dumps(server)})
        samples=[]
        for i in range(1,4):
            response={'status':200, 'body':json.dumps([{'meta_info':{'prompt_tokens':2,'completion_tokens':64,'cached_tokens':0}}]*128)}
            record={'kind':'measured','iteration':i,'flush':{'status':200,'body':'Cache flushed.\n'},
                    'request':M.capture.generation_request(self.frozen,f'{self.capture}-{mode}-{i}'), 'response':response,
                    'metrics':M.wrapper.batch_metrics(response,128,i+.5)}
            self.write(folder/f'measured-{i}.json',record);samples.append(record['metrics'])
        profile_id=self.capture+'-'+mode
        self.write(folder/'capture-claim.json', {'engine':engine,'payload':M.capture.stage_payload('/run-output/diagnostic/'+mode+'/traces',profile_id)})
        self.write(folder/'capture-request.json', {**record,'arm':{'status':200},
            'request':M.capture.generation_request(self.frozen,f'{self.capture}-{mode}-profile')})
        trace=folder/'traces'/(profile_id+'-TP-0-DECODE.trace.json.gz');trace.parent.mkdir()
        with gzip.open(trace,'wt') as stream:json.dump(document(mode),stream)
        evidence=M.capture.inspect_trace(trace)
        self.write(folder/'summary.json',{'mode':mode,'engine_identity':engine,'measurements':samples,'capture':{'trace_evidence':[evidence]}})

    def seal(self):
        self.write(self.root/'diagnostic-retention.json', {'source_before_after_and_destination_verified':True,
            'files':{str(p.relative_to(self.node)):{'bytes':p.stat().st_size,'sha256':M.sha(p)} for p in self.node.rglob('*') if p.is_file()}})

    def run_build(self):return M.build(self.experiment,self.inputs,self.base/'generated')

    def test_maps_raw_http_samples_and_actual_mixed_stages(self):
        result=self.run_build();self.assertEqual(result['verified_pairs'],1);self.assertEqual(result['verified_profiles'],2)
        diagnostic=M.load(self.base/'generated/diagnostics.json')['pairs'][0]
        self.assertEqual(diagnostic['on']['generation_seconds'],[1.5,2.5,3.5])
        self.assertEqual(diagnostic['on']['request_sha256'],diagnostic['off']['request_sha256'])
        self.assertNotEqual(diagnostic['on']['literal_request_sha256'],diagnostic['off']['literal_request_sha256'])
        profiles=M.load(self.base/'generated/profiles.json')
        self.assertEqual(profiles['graph_evidence'][0]['decode_graph_replays'],2)
        self.assertEqual([x['stage'] for x in profiles['profiles']],['prefill','decode'])
        decode=profiles['profiles'][1]
        self.assertEqual(decode['selection']['analyzer_forward_index'],1) # Earliest, not faster index2.
        self.assertAlmostEqual(decode['selection']['start_ms_relative_to_first_device_gpu_event'],.195)
        self.assertEqual(decode['source_trace_sha256'],profiles['profiles'][0]['source_trace_sha256'])
        spec=importlib.util.spec_from_file_location('deck_guard',ROOT/'generate.py');G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
        G.build_report(experiment=self.experiment,profiles=self.base/'generated/profiles.json',
                       diagnostics=self.base/'generated/diagnostics.json',output=self.base/'schema-site')

    def test_retained_tamper_rejected_before_output(self):
        path=self.node/'diagnostic/on/measured-1.json';path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.run_build()
        self.assertFalse((self.base/'generated').exists())

    def test_wrong_main_or_capture_rejected(self):
        path=self.root/'recovery-plan.json';data=M.load(path);data['config']['run_id']='OTHER_MAIN';self.write(path,data)
        with self.assertRaisesRegex(ValueError,'identity mismatch'):self.run_build()

    def test_raw_response_not_just_summary_is_validated(self):
        p=self.node/'diagnostic/on/measured-1.json';data=M.load(p);data['metrics']['output_tokens']=1;self.write(p,data);self.seal()
        with self.assertRaisesRegex(ValueError,'response/measurement'):self.run_build()

    def test_changed_request_or_context_rejected(self):
        p=self.node/'diagnostic/on/measured-1.json';data=M.load(p);data['request']['request']['input_ids'][0]=[999];self.write(p,data);self.seal()
        with self.assertRaisesRegex(ValueError,'Frozen request'):self.run_build()

    def test_missing_off_is_pending_but_on_trace_survives(self):
        (self.node/'diagnostic/off/summary.json').unlink();self.seal();result=self.run_build()
        self.assertEqual(result['verified_pairs'],0);self.assertEqual(result['verified_profiles'],2)

    def test_absent_decode_proof_stays_unknown_and_pending(self):
        p=next((self.node/'diagnostic/on/traces').glob('*.gz'));doc=document('off')
        with gzip.open(p,'wt') as stream:json.dump(doc,stream)
        summary_path=self.node/'diagnostic/on/summary.json';summary=M.load(summary_path)
        summary['capture']['trace_evidence']=[M.capture.inspect_trace(p)];self.write(summary_path,summary);self.seal()
        result=self.run_build();self.assertEqual(result['verified_profiles'],1);self.assertEqual(result['verified_pairs'],0)
        graph=M.load(self.base/'generated/profiles.json')['graph_evidence'][0]
        self.assertEqual((graph['decode_unknown'],graph['decode_fallbacks']),(2,0))

    def test_missing_summaries_and_failed_terminal_stay_pending(self):
        for mode in ['off','on']:(self.node/'diagnostic'/mode/'summary.json').unlink()
        self.write(self.node/'diagnostic/terminal.json',{'status':'FAILED','error':'synthetic failure'})
        self.seal();result=self.run_build()
        self.assertEqual(result['profile_status'],'pending');self.assertEqual(result['verified_pairs'],0)
        evidence=M.load(self.base/'generated/rubin/evidence.json')
        self.assertEqual(evidence['wrapper_terminal']['status'],'FAILED')

    def test_request_from_another_capture_is_rejected(self):
        p=self.node/'diagnostic/on/measured-1.json';data=M.load(p)
        data['request']=M.capture.generation_request(self.frozen,'OTHER_CAPTURE-on-1')
        self.write(p,data);self.seal()
        with self.assertRaisesRegex(ValueError,'Frozen request'):self.run_build()

    def test_existing_output_refused(self):
        (self.base/'generated').mkdir()
        with self.assertRaisesRegex(ValueError,'Fresh output'):self.run_build()


if __name__=='__main__':unittest.main()
