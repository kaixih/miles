#!/usr/bin/env python3
"""Run two bounded BF16 MoE checks on one already-requested Rubin allocation.

Execute on the login host as UID28644:GID30. The prepared --root must contain
inputs/train.jsonl and inputs/test-fixed-256.jsonl. This utility neither submits
nor cancels a Slurm job. It uses a fresh output/cache namespace for each backend,
keeps initial evaluation enabled, runs two rollouts, and writes no checkpoints
or tensor dumps. No background model copy overlaps timed work.
"""

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback


BASE_IMAGE = "gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin@sha256:a03106bdd90c5d6067fbff246fff25df979f9da8486eb0dac795a315a2346d6c"
UID, GID = 28644, 30
INPUT_HASHES = {
    "train.jsonl": "f5ca349cacea3a32998ccd59fae4ecd0007bcec1bd26c9ad16d732fad1a369d8",
    "test-fixed-256.jsonl": "93ed3ccda6ecd09ce0665d0423bf8720b7db97823b0ce2a1c4d20483f410ce99",
}
MODELS = ("Qwen3-30B-A3B", "Qwen3-30B-A3B_torch_dist")
ALLREDUCE = '''import datetime, os, torch
import torch.distributed as dist
rank=int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120))
assert dist.get_world_size()==4
for size in (1,4096,1048576):
    value=torch.full((size,),float(rank+1),device="cuda")
    dist.all_reduce(value)
    torch.cuda.synchronize()
    assert torch.all(value==10).item()
dist.barrier()
if rank==0: print("ALLREDUCE_OK world=4",flush=True)
dist.destroy_process_group()
'''


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def summarize_log(path):
    """Keep small metrics only; never copy tokens or tensors into this receipt."""
    result = {"generation": {}, "stages": {}, "train_steps": {}, "normal_rank_steps": [],
              "invalid_rank_outcomes": [], "metric_conflicts": [], "effective_backends": [],
              "decode_graph_capture_count": 0, "decode_graph_true_samples": 0,
              "decode_graph_false_samples": 0, "submission_id": None}
    if not path.exists():
        return result
    for line in path.open(errors="replace"):
        found = re.search(r"'moe_runner_backend': '([^']+)'", line)
        if found and found.group(1) not in result['effective_backends']:
            result['effective_backends'].append(found.group(1))
        result['decode_graph_capture_count'] += int('Capture target decode CUDA graph end.' in line)
        result['decode_graph_true_samples'] += int('Decode batch,' in line and 'cuda graph: True' in line)
        result['decode_graph_false_samples'] += int('Decode batch,' in line and 'cuda graph: False' in line)
        found = re.search(r'actor_cell0_rank(\d+)\].*?op=train_step rollout=(\d+) step=(\d+) attempt=(\d+) outcome=(\w+) valid_step=(\w+)',line)
        if found:
            rank,rollout,step,attempt,outcome,valid=found.groups()
            if outcome=='NORMAL' and valid=='true':
                key=[int(rank),int(rollout),int(step)]
                if key not in result['normal_rank_steps']:result['normal_rank_steps'].append(key)
            else:result['invalid_rank_outcomes'].append(found.groups())
        found = re.search(r' - step ([0-7]): (\{.*\})',line)
        if found:
            try:row=ast.literal_eval(found.group(2))
            except (ValueError,SyntaxError):row={}
            if 'train/grad_norm' in row:
                previous=result['train_steps'].get(found.group(1))
                if previous is not None and previous!=row:result['metric_conflicts'].append(found.group(1))
                result['train_steps'][found.group(1)]=row
        found = re.search(r"raysubmit_[A-Za-z0-9_-]+", line)
        if found:
            result["submission_id"] = found.group(0)
        found = re.search(r"perf ([01]): (\{.*\})", line)
        if found:
            try:
                row = ast.literal_eval(found.group(2))
            except (ValueError, SyntaxError):
                continue
            section = "generation" if "perf/rollout_time" in row else "stages"
            if section == "stages" and "perf/step_time" not in row:
                continue
            result[section][found.group(1)] = row
            if section == "generation":
                row["generated_tokens"] = round(row["rollout/num_training_samples"] * row["rollout/response_len/mean"])
    return result


class Runner:
    def __init__(self, args):
        self.a = args
        self.root = args.root.resolve(strict=True)
        self.source = args.source.resolve(strict=True)
        self.context = args.build_context.resolve(strict=True)
        if (os.getuid(),os.getgid())!=(UID,GID) or self.root.stat().st_uid!=UID:
            raise RuntimeError('Run from the prepared root as host UID28644:GID30')
        self.ops = self.root / "ops"
        self.ops.mkdir(exist_ok=True)
        self.deadline = None
        self.lease_end = None
        self.node = None
        self.active = None
        self.sequence = 0
        self.receipt = {"started_utc": utc(), "job_id": args.job_id,
                        "root": str(self.root), "source": str(self.source),
                        "base_image": args.base_image, "backends": {}, "stage": "prepared"}

    def state(self, stage, **extra):
        self.receipt.update(stage=stage, observed_utc=utc(), **extra)
        write_json(self.root / "state.json", self.receipt)
        print(json.dumps({"utc": utc(), "stage": stage, **extra}, default=str), flush=True)

    def budget(self, seconds, cleanup=False):
        boundary = self.lease_end if cleanup else self.deadline
        if boundary is not None:
            seconds = min(seconds, boundary - time.time() - 5)
        if seconds <= 0:
            raise TimeoutError("Absolute allocation deadline reached")
        return max(1, int(seconds))

    def command(self, argv, *, timeout=300, log=None, check=True, cleanup=False):
        timeout = self.budget(timeout, cleanup)
        self.sequence += 1
        log = log or self.ops / f"command-{self.sequence:03d}.log"
        with (self.ops / "commands.jsonl").open("a") as stream:
            stream.write(json.dumps({"utc": utc(), "argv": argv, "timeout_s": timeout,
                                     "log": str(log)}) + "\n")
        with log.open("w") as stream:
            process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                       text=True, start_new_session=True, env={**os.environ, "TZ": "UTC"})
            stop = time.monotonic() + timeout
            try:
                while process.poll() is None:
                    if time.monotonic() >= stop:
                        raise TimeoutError(f"Command timed out: {argv[0]}; see {log}")
                    if log.stat().st_size > 512 * 1024**2:
                        raise RuntimeError(f"Command log exceeded 512MiB: {log}")
                    time.sleep(min(2, max(0.05, stop - time.monotonic())))
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        if check and process.returncode:
            raise RuntimeError(f"Command exited {process.returncode}: {argv[0]}; see {log}")
        # Large training/build logs stay on disk. Callers needing JSON use small commands.
        return process.returncode, log.read_text(errors="replace") if log.stat().st_size < 16 * 1024**2 else ""

    def job(self):
        _, text = self.command(["scontrol", "show", "job", "-o", self.a.job_id], timeout=30, cleanup=True)
        fields = dict(item.split("=", 1) for item in text.split() if "=" in item)
        if fields.get("JobId") != self.a.job_id or f"({UID})" not in fields.get("UserId", ""):
            raise RuntimeError("Allocation ID or owner differs from requested job")
        return fields

    def require_own_node(self):
        fields = self.job()
        if fields.get("JobState") != "RUNNING":
            raise RuntimeError(f"Own allocation no longer RUNNING: {fields.get('JobState')}")
        _, text = self.command(["scontrol", "show", "hostnames", fields["NodeList"]], timeout=30, cleanup=True)
        if text.split() != [self.node]:
            raise RuntimeError("Own allocation node changed")
        current_end=dt.datetime.fromisoformat(fields['EndTime']).replace(tzinfo=dt.timezone.utc).timestamp()
        if self.lease_end is not None:
            # A shortened lease applies immediately; an extension cannot silently
            # expand the original experiment budget.
            self.lease_end=min(self.lease_end,current_end)
            self.deadline=min(self.deadline,self.lease_end-15*60)

    def remote(self, argv, *, timeout=300, log=None, check=True, cleanup=False):
        self.require_own_node()
        limit = self.budget(timeout, cleanup)
        command = ["timeout", "--kill-after=10s", f"{max(1,limit-1)}s", *argv]
        return self.command(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
                             self.node, shlex.join(command)], timeout=limit, log=log,
                            check=check, cleanup=cleanup)

    def wait_allocation(self):
        stop = time.monotonic() + min(self.a.queue_timeout_seconds, 4 * 3600)
        while True:
            fields = self.job()
            if fields.get("JobState") == "RUNNING":
                break
            if fields.get("JobState") not in {"PENDING", "CONFIGURING"}:
                raise RuntimeError(f"Own allocation cannot start: {fields.get('JobState')}")
            if time.monotonic() >= stop:
                raise TimeoutError("Own allocation did not start within four-hour queue budget")
            self.state("waiting_for_own_allocation", allocation=fields)
            time.sleep(min(30, max(0, stop-time.monotonic())))
        _, text = self.command(["scontrol", "show", "hostnames", fields["NodeList"]], timeout=30)
        nodes = text.split()
        if len(nodes) != 1 or "gres/gpu=4" not in fields.get("AllocTRES", ""):
            raise RuntimeError("Require exactly one allocated four-GPU node")
        self.node = nodes[0]
        self.lease_end = dt.datetime.fromisoformat(fields["EndTime"]).replace(tzinfo=dt.timezone.utc).timestamp()
        self.deadline = self.lease_end - 15 * 60
        if self.lease_end - time.time() < 60 * 60:
            raise RuntimeError("Require at least sixty minutes of allocation remaining")
        self.state("allocation_ready", node=self.node, allocation=fields,
                   lease_end_utc=dt.datetime.fromtimestamp(self.lease_end,dt.timezone.utc).isoformat(),
                   deadline_utc=dt.datetime.fromtimestamp(self.deadline,dt.timezone.utc).isoformat())

    def storage_and_models(self):
        self.state("storage_preflight")
        slug = re.sub(r"[^a-zA-Z0-9_.-]", "-", self.root.name)[:65]
        local_name = f"miles-kaixih-j{self.a.job_id}-moe-{slug}"
        config = {"root":str(self.root), "local_name":local_name, "uid":UID, "gid":GID}
        code = '''import json,os,pathlib,shutil,subprocess,tempfile
c=json.loads(CONFIG)
assert (os.getuid(),os.getgid())==(c['uid'],c['gid'])
root=pathlib.Path(c['root']); assert root.is_dir() and root.stat().st_uid==c['uid']
mount=subprocess.check_output(['findmnt','-T',str(root),'-n','-o','SOURCE,FSTYPE,TARGET'],text=True)
assert 'nfs' in mount.lower(),mount
base=None
for candidate in ['/raid/dldata','/tmp']:
    p=pathlib.Path(candidate)
    if not p.is_dir() or not os.access(p,os.W_OK) or shutil.disk_usage(p).free<=300*1024**3:continue
    kind=subprocess.check_output(['findmnt','-T',candidate,'-n','-o','FSTYPE'],text=True).strip()
    if kind in {'ext4','xfs','btrfs','zfs'}:base=p;break
assert base is not None,'Require writable node-local disk with >300GiB free; no parent permissions changed'
local=base/c['local_name']; assert not local.exists(),'Fresh local namespace required'
local.mkdir(mode=0o700)
for child in ['models','probe','cutlass-cache','trtllm-cache']:(local/child).mkdir()
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=name,compute_cap','--format=csv,noheader'],text=True).strip().splitlines()
assert len(gpus)==4 and all('10.7' in x and 'VR' in x for x in gpus),gpus
interfaces=json.loads(subprocess.check_output(['ip','-j','address','show','mp0'],text=True))
ips=[a['local'] for i in interfaces for a in i.get('addr_info',[]) if a['family']=='inet' and a.get('scope')=='global']
assert len(ips)==1,ips
print(json.dumps({'node_ip':ips[0],'gpus':gpus,'mount':mount,'local_root':str(local),'local_mount':subprocess.check_output(['findmnt','-T',str(local),'-n','-o','SOURCE,FSTYPE,TARGET'],text=True),'uid':os.getuid(),'gid':os.getgid()}))
'''.replace("CONFIG", repr(json.dumps(config)))
        _, out = self.remote(["python3","-c",code], timeout=60)
        storage = json.loads(out.strip().splitlines()[-1]); self.ip=storage["node_ip"];self.local=storage['local_root']
        write_json(self.root / "storage.json", storage)
        _,out=self.remote(['nvidia-smi','-q','-x'],timeout=30)
        write_json(self.root/'hardware.json',{'observed_utc':utc(),'node':self.node,'nvidia_smi_xml':out})
        self.require_idle_gpus()
        self.remote(["docker","pull",self.a.base_image], timeout=1200, log=self.ops/"base-image-pull.log")
        probe = self.root / "container-write-probe"
        probe_code = '''import os,pathlib,json
assert (os.getuid(),os.getgid())==(28644,30)
p=pathlib.Path('/run-output/container-write-probe');p.mkdir();(p/'nested').mkdir();(p/'nested'/'ok').write_text('ok')
q=pathlib.Path('/cache/inside');q.mkdir();(q/'ok').write_text('ok')
print(json.dumps({'uid':os.getuid(),'gid':os.getgid()}))
'''
        self.remote(["docker","run","--rm","--name",f"miles-moe-probe-j{self.a.job_id}-{slug}",
                     "--user",f"{UID}:{GID}","--mount",f"type=bind,src={self.root},dst=/run-output",
                     "--mount",f"type=bind,src={self.local}/probe,dst=/cache",self.a.base_image,
                     "python3","-c",probe_code], timeout=120)
        p=probe/"nested"/"ok"
        if p.read_text()!="ok" or (p.stat().st_uid,p.stat().st_gid)!=(UID,GID):
            raise RuntimeError("Container shared-output ownership/readback failed")
        p.unlink();p.parent.rmdir();probe.rmdir()
        self.remote(["python3","-c",f"from pathlib import Path;p=Path({self.local+'/probe/inside'!r});f=p/'ok';assert f.read_text()=='ok' and (f.stat().st_uid,f.stat().st_gid)==({UID},{GID});f.unlink();p.rmdir()"],timeout=30)
        self.state("staging_models", storage_preflight="passed", local_root=self.local)
        for name in MODELS:
            source = str(self.a.models / name)
            destination = f"{self.local}/models/{name}"
            self.remote(["rsync","-rlt","--chmod=Du+rwx","--",source,f"{self.local}/models/"],timeout=1800,
                        log=self.ops/f"stage-{name}.log")
            code = '''import hashlib,json,pathlib
source=pathlib.Path(SOURCE);target=pathlib.Path(TARGET)
def inventory(p):
    return {str(x.relative_to(p)):x.stat().st_size for x in p.rglob('*') if x.is_file()}
a=inventory(source);b=inventory(target);assert a==b and sum(a.values())>50*1024**3
for name in ['config.json','model.safetensors.index.json','tokenizer_config.json','latest_checkpointed_iteration.txt','release/.metadata']:
    x=source/name
    if x.is_file():assert hashlib.sha256(x.read_bytes()).digest()==hashlib.sha256((target/name).read_bytes()).digest()
print(json.dumps({'source':str(source),'target':str(target),'files':len(a),'bytes':sum(a.values()),'inventory_sha256':hashlib.sha256(json.dumps(a,sort_keys=True).encode()).hexdigest()}))
'''.replace("SOURCE",repr(source)).replace("TARGET",repr(destination))
            _,out=self.remote(["python3","-c",code],timeout=180)
            write_json(self.root/f"model-{name}.json",json.loads(out.strip().splitlines()[-1]))

    def orchestrator(self, action, backend, image, output, prefix):
        launcher = ["--run-id",f"{self.root.name}-{backend}","--sglang-moe-runner-backend",backend,
                    "--sglang-enable-cuda-graph","--max-tokens-per-gpu","4096",
                    "--rollout-batch-size","256","--n-samples-per-prompt","8","--global-batch-size","512",
                    "--rollout-max-prompt-len","512","--rollout-max-response-len","1024",
                    "--save-interval","0","--save-retain-interval","0",
                    "--prompt-data-path","/run-output/inputs/train.jsonl",
                    "--eval-prompt-data-path","/run-output/inputs/test-fixed-256.jsonl"]
        cache_name="cutlass-cache" if backend=="flashinfer_cutlass" else "trtllm-cache"
        return [sys.executable,"-u",str(self.source/"lab/rubin_two_node/orchestrate_rubin.py"),action,
                "--nodes",self.node,"--node-ips",self.ip,"--job-id",self.a.job_id,
                "--uid",str(UID),"--gid",str(GID),"--image",image,"--repo",str(self.source),
                "--node-repo",str(self.source),"--models",str(self.a.models),
                "--node-models",f"{self.local}/models","--run-dir",str(output),
                "--cache-dir",f"{self.local}/{cache_name}","--container-prefix",prefix,
                "--ray-port","26379","--dashboard-port","28265","--recipe","qwen3-gsm8k",
                "--num-rollout","2","--launcher-args",shlex.join(launcher),
                "--gate-log",str(output/"allreduce.log")]

    def cleanup(self):
        if self.active is None:
            return
        name, output = self.active
        code,out=self.remote(["docker","inspect",name],timeout=30,check=False,cleanup=True)
        if code:
            if "No such object" in out or "No such container" in out:
                self.active=None;return
            raise RuntimeError(f"Cannot inspect own container for cleanup: {name}")
        record=json.loads(out)[0]
        mounts={m['Destination']:m['Source'] for m in record['Mounts']}
        if record['Name']!='/'+name or mounts.get('/run-output')!=str(output) or record['Config']['User']!=f'{UID}:{GID}':
            raise RuntimeError("Refusing cleanup of container with unexpected identity/mounts")
        self.remote(["docker","stop","--time","15",name],timeout=40,cleanup=True)
        self.remote(["docker","rm",name],timeout=30,cleanup=True)
        self.active=None

    def require_idle_gpus(self):
        _,out=self.remote(['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid','--format=csv,noheader'],timeout=30)
        if out.strip():raise RuntimeError('Unexpected GPU processes; no foreign cleanup attempted: '+out[:2000])

    def backend(self, backend, image):
        self.require_own_node()
        self.require_idle_gpus()
        output=self.root/backend;output.mkdir()
        (output/"inputs").mkdir()
        for name in INPUT_HASHES:shutil.copyfile(self.root/"inputs"/name,output/"inputs"/name)
        (output/"allreduce.py").write_text(ALLREDUCE)
        prefix=f"miles-moe-j{self.a.job_id}-{hashlib.sha256(str(self.root).encode()).hexdigest()[:8]}-{backend}"
        name=prefix+"-0"
        code,inspection=self.remote(["docker","container","inspect",name],check=False,timeout=30)
        if code==0:raise RuntimeError("Own unique container name already exists; refusing reuse")
        if 'No such object' not in inspection and 'No such container' not in inspection:
            raise RuntimeError("Could not establish that own container name is unused")
        self.active=(name,output)
        result={"backend":backend,"image":image,"started_utc":utc(),"output":str(output)}
        self.receipt["backends"][backend]=result
        try:
            self.state("bootstrap",backend=backend)
            self.command(self.orchestrator("bootstrap",backend,image,output,prefix),timeout=600,log=output/"bootstrap.log")
            code='''import importlib.metadata,json,os,pathlib,torch
versions={}
for name in ['torch','sglang','flashinfer-python','triton','transformer-engine','ray']:
    try:versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:versions[name]=None
p=pathlib.Path('/opt/miles-moe/provenance.json')
print(json.dumps({'uid':os.getuid(),'gid':os.getgid(),'cuda':torch.version.cuda,'versions':versions,'gpu_count':torch.cuda.device_count(),'refit_provenance':json.loads(p.read_text()) if p.exists() else None}))
'''
            _,out=self.remote(['docker','exec',name,'python3','-c',code],timeout=90)
            write_json(output/'runtime-provenance.json',json.loads(out.strip().splitlines()[-1]))
            self.remote(["docker","exec",name,"python3","-m","torch.distributed.run","--nnodes=1","--nproc-per-node=4",
                         "--master-addr=127.0.0.1","--master-port=29500","/run-output/allreduce.py"],timeout=180,log=output/"allreduce.log")
            self.state("training",backend=backend)
            train_code,_=self.command(self.orchestrator("train",backend,image,output,prefix),timeout=1500,
                                      log=output/"orchestrator-train.log",check=False)
            metrics=summarize_log(output/"logs/qwen3_train.log")
            write_json(output/"metrics.json",metrics)
            sid=metrics["submission_id"]
            if not sid:raise RuntimeError("Training log did not identify a Ray submission")
            ray=self.ray_status(name,output,sid)
            result.update(train_exit_code=train_code,submission_id=sid,ray_status=ray.get("status"))
            if train_code or ray.get("status")!="SUCCEEDED":raise RuntimeError("Training or actual Ray job did not succeed")
            if set(metrics["generation"])!={"0","1"} or set(metrics["stages"])!={"0","1"}:
                raise RuntimeError("Missing generation/stage records for both rollouts")
            for i in (0,1):
                row=metrics['generation'][str(i)]
                if row.get('rollout/weight_version/mean')!=i+1 or row.get('rollout/weight_version/mixed_version_ratio')!=0:
                    raise RuntimeError("Weight version did not advance cleanly")
            expected={(rank,r,s) for rank in range(4) for r in range(2) for s in range(4)}
            finite=lambda value: type(value) in (float,int) and math.isfinite(value)
            checks={
                'eight_optimizer_steps':set(metrics['train_steps'])=={str(i) for i in range(8)} and all(row.get('train/step')==int(i) for i,row in metrics['train_steps'].items()),
                'finite_positive_gradients':bool(metrics['train_steps']) and all(finite(x.get('train/grad_norm')) and x['train/grad_norm']>0 for x in metrics['train_steps'].values()),
                'finite_losses':bool(metrics['train_steps']) and all(finite(x.get('train/loss')) for x in metrics['train_steps'].values()),
                'all_32_rank_steps_normal':{tuple(x) for x in metrics['normal_rank_steps']}==expected and not metrics['invalid_rank_outcomes'],
                'no_metric_conflicts':not metrics['metric_conflicts'],
                'effective_backend_matches':metrics['effective_backends']==[backend],
                'decode_capture_and_replay_logged':metrics['decode_graph_capture_count']>=4 and metrics['decode_graph_true_samples']>0,
            }
            result['checks']=checks
            if not all(checks.values()):raise RuntimeError('Incomplete execution validation: '+str(checks))
            result['status']='SUCCEEDED'
        except Exception as exc:
            result.update(status="FAILED",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
        finally:
            metrics=summarize_log(output/"logs/qwen3_train.log")
            write_json(output/"metrics.json",metrics)
            if metrics['submission_id'] and result.get('status')!='SUCCEEDED':
                try:
                    ray=self.ray_status(name,output,metrics['submission_id'],stop_if_active=True)
                    result.update(submission_id=metrics['submission_id'],ray_status=ray.get('status'))
                except Exception as exc:
                    result['ray_terminal_error']=str(exc)
            result['finished_utc']=utc();write_json(output/"result.json",result)
            self.state("backend_finished",backend=backend)
            self.cleanup()  # Cleanup failure stops the sequence; never overlap backends.

    def ray_status(self,name,output,sid,stop_if_active=False):
        code="""import json,time
from ray.job_submission import JobSubmissionClient
c=JobSubmissionClient(ADDRESS);v=c.get_job_info(SUBMISSION)
terminal={'SUCCEEDED','FAILED','STOPPED'}
if STOP_IF_ACTIVE and v.status not in terminal:
    c.stop_job(SUBMISSION)
    deadline=time.monotonic()+30
    while v.status not in terminal and time.monotonic()<deadline:
        time.sleep(2);v=c.get_job_info(SUBMISSION)
out={key:getattr(v,key,None) for key in ['status','message','error_type','start_time','end_time','exit_code']}
out['submission_id']=SUBMISSION;out['terminal_observed']=v.status in terminal
print(json.dumps(out,default=str))
""".replace("ADDRESS",repr(f"http://{self.ip}:28265")).replace("SUBMISSION",repr(sid)).replace("STOP_IF_ACTIVE",repr(stop_if_active))
        _,out=self.remote(["docker","exec",name,"python3","-c",code],timeout=45,cleanup=stop_if_active)
        ray=json.loads(out.strip().splitlines()[-1]);write_json(output/"ray-terminal.json",ray)
        return ray

    def execute(self):
        if (os.getuid(),os.getgid())!=(UID,GID):raise RuntimeError("Run as host UID28644:GID30")
        if self.a.base_image!=BASE_IMAGE:raise ValueError("This experiment requires the recorded base image digest")
        for name,digest in INPUT_HASHES.items():
            if hashlib.sha256((self.root/"inputs"/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f"Unexpected dataset bytes: {name}")
        if not (self.context/"Dockerfile.trtllm-refit").is_file():raise FileNotFoundError("Missing derivative Dockerfile")
        (self.root/"runner.lock").mkdir()  # Existing lock/output requires explicit review, never implicit resume.
        write_json(self.root/'runner.lock'/'identity.json',{'pid':os.getpid(),'uid':os.getuid(),'created_utc':utc(),'job_id':self.a.job_id})
        try:
            self.wait_allocation();self.storage_and_models()
            self.backend("flashinfer_cutlass",self.a.base_image)
            self.state("building_trtllm_derivative")
            tag=f"miles-rubin-trtllm-refit:j{self.a.job_id}-{hashlib.sha256(str(self.root).encode()).hexdigest()[:8]}"
            self.remote(["env","DOCKER_BUILDKIT=1","docker","build","--network=none","--build-arg",f"BASE_IMAGE={self.a.base_image}",
                         "--file",str(self.context/"Dockerfile.trtllm-refit"),"--tag",tag,str(self.context)],
                        timeout=600,log=self.root/"trtllm-image-build.log")
            _,out=self.remote(["docker","image","inspect",tag],timeout=30)
            image_info=json.loads(out)
            write_json(self.root/"trtllm-image.json",image_info)
            self.backend("flashinfer_trtllm",image_info[0]['Id'])
            if self.receipt['backends']['flashinfer_trtllm'].get('status')=='SUCCEEDED':
                self.push_image(image_info[0]['Id'])
            success=all(x.get('status')=='SUCCEEDED' for x in self.receipt['backends'].values())
            self.state("completed" if success else "completed_with_failures",finished_utc=utc())
            return 0 if success else 1
        except BaseException as exc:
            self.state("failed",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
            try:self.cleanup()
            except Exception as cleanup_error:self.state("failed_cleanup",cleanup_error=str(cleanup_error))
            raise

    def push_image(self,image_id):
        target=self.a.registry_tag or f'gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin:experimental-moe-pr33743-cu134-20260916-j{self.a.job_id}'
        result={'image_id':image_id,'target':target,'started_utc':utc()}
        self.state('pushing_validated_derivative',registry=result)
        try:
            self.remote(['docker','tag',image_id,target],timeout=30)
            self.remote(['docker','push',target],timeout=180,log=self.root/'registry-push.log')
            _,out=self.remote(['docker','image','inspect',target],timeout=30)
            info=json.loads(out)[0]
            result.update(status='PUSHED',repo_digests=info.get('RepoDigests',[]),finished_utc=utc())
        except Exception as exc:
            result.update(status='PUSH_FAILED',error=str(exc),finished_utc=utc())
        write_json(self.root/'registry-push.json',result)
        self.receipt['registry']=result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id',required=True)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--base-image',default=BASE_IMAGE)
    parser.add_argument('--build-context',required=True,type=Path)
    parser.add_argument('--models',type=Path,default=Path('/home/scratch.kaixih_ent/models'))
    parser.add_argument('--queue-timeout-seconds',type=int,default=4*3600)
    parser.add_argument('--registry-tag',help='Optional unique tag under the existing kaixih/my_docker_hub/miles-rubin registry')
    args=parser.parse_args()
    if not args.job_id.isdigit():parser.error('--job-id must be a numeric existing allocation ID')
    if not 1<=args.queue_timeout_seconds<=4*3600:parser.error('Queue timeout must be 1..14400 seconds')
    if args.registry_tag and not args.registry_tag.startswith('gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin:'):
        parser.error('--registry-tag must remain in the existing miles-rubin registry')
    return Runner(args).execute()


if __name__=='__main__':
    raise SystemExit(main())
