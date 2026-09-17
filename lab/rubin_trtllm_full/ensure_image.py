#!/usr/bin/env python3
"""Prepare only this campaign's image on its allocated node; run on dl3."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

BASE = Path('/home/scratch.kaixih_ent/repro/miles-qwen3-trtllm-full/20260917-campaign')
JOBS = {'rubin': '2212643', 'gb300': '2212644'}
RUBIN = 'gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-rubin@sha256:405315ba3add773be16cfe176a4dcd15a1071cbf17f3be255ab2c47324c65f34'
UPSTREAM = 'radixark/miles@sha256:226f63d28e4b1482e0a6948ba3d486c1b1635648d079c82c9501640b24657986'
TARGET = 'gitlab-master.nvidia.com:5005/kaixih/my_docker_hub/miles-gb300:experimental-trtllm-refit-cu130-20260917-j2212644'


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(tmp, path)


def allocation(job):
    raw = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True, timeout=30)
    rec = dict(x.split('=', 1) for x in raw.split() if '=' in x)
    assert rec['JobId'] == job and rec['UserId'] == 'kaixih(28644)'
    assert rec['JobState'] == 'RUNNING' and rec['NodeList'] and ',' not in rec['NodeList'] and '[' not in rec['NodeList']
    end = dt.datetime.fromisoformat(rec['EndTime']).replace(tzinfo=dt.timezone.utc).timestamp()
    assert end - time.time() > 3600, 'Need one hour of existing lease after image work'
    return rec


class Image:
    def __init__(self, platform):
        self.platform = platform
        self.job = JOBS[platform]
        self.lease = allocation(self.job)
        self.root = BASE / (platform + '-image')
        self.root.mkdir(exist_ok=True)

    def remote(self, argv, timeout=60, log=None):
        current = allocation(self.job)
        assert all(current[k] == self.lease[k] for k in ('NodeList', 'StartTime', 'EndTime'))
        argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', current['NodeList'],
                shlex.join(['timeout', '--signal=TERM', '--kill-after=20s', str(timeout) + 's', *argv])]
        if log:
            path = self.root / log
            with path.open('a') as f:
                f.write('\n# ' + utc() + ' ' + shlex.join(argv) + '\n'); f.flush()
                subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, timeout=timeout + 45, check=True)
            return None
        return subprocess.check_output(argv, text=True, timeout=timeout + 45)

    def inspect(self, ref):
        return json.loads(self.remote(['docker', 'image', 'inspect', ref]))[0]

    def run(self, verify_existing=False):
        receipt = {'platform': self.platform, 'job': self.job, 'allocation': self.lease, 'started_utc': utc()}
        write(self.root / 'started.json', receipt)
        if self.platform == 'rubin':
            self.remote(['docker', 'pull', RUBIN], timeout=1800, log='pull.log')
            ref = RUBIN
        else:
            context = BASE / 'gb300-image-context'
            receipt['build_context_sha256'] = {
                str(p.relative_to(context)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(context.rglob('*')) if p.is_file()}
            write(self.root / 'build-context.json', receipt['build_context_sha256'])
            try:
                self.inspect(UPSTREAM)
            except subprocess.CalledProcessError:
                self.remote(['docker', 'pull', UPSTREAM], timeout=1800, log='base-pull.log')
            if not verify_existing:
                self.remote(['env', 'DOCKER_BUILDKIT=1', 'docker', 'build', '--pull=false', '--network=none',
                             '--progress=plain', '--tag', TARGET, '/mnt/cifs' + str(context)],
                            timeout=1800, log='build.log')
            built = self.inspect(TARGET)
            receipt['built_image_id'] = built['Id']
            self.remote(['docker', 'push', TARGET], timeout=1200, log='push.log')
            inspected = self.inspect(TARGET)
            repository = TARGET.rsplit(':', 1)[0]
            refs = [x for x in inspected.get('RepoDigests', []) if x.startswith(repository + '@sha256:')]
            assert len(refs) == 1, 'Require unique pushed image digest'
            ref = refs[0]
            registry = json.loads(self.remote(['docker', 'manifest', 'inspect', ref], timeout=120))
            write(self.root / 'registry-manifest.json', registry)
            if 'manifests' in registry:
                manifests = [m for m in registry['manifests']
                             if m.get('platform', {}).get('os') == 'linux'
                             and m.get('platform', {}).get('architecture') == 'arm64']
                assert len(manifests) == 1, 'Require one actual ARM64 platform manifest'
                platform_ref = repository + '@' + manifests[0]['digest']
                registry = json.loads(self.remote(['docker', 'manifest', 'inspect', platform_ref], timeout=120))
                write(self.root / 'registry-arm64-manifest.json', registry)
                receipt['platform_manifest'] = platform_ref
            receipt['registry_config_digest'] = registry['config']['digest']
            if built.get('Descriptor', {}).get('mediaType') == 'application/vnd.oci.image.index.v1+json':
                assert built['Descriptor']['digest'] == ref.split('@')[1] == built['Id'], 'Registry index differs from tested image'
            else:
                assert registry['config']['digest'] == built['Id'], 'Registry config differs from tested image'
            self.remote(['docker', 'pull', ref], timeout=300, log='readback.log')
            readback = self.inspect(ref)
            assert readback['Id'] == built['Id'] and readback['RootFS'] == built['RootFS']
        final = self.inspect(ref)
        receipt.update(image=ref, image_id=final['Id'], architecture=final['Architecture'],
                       repo_digests=final.get('RepoDigests'), finished_utc=utc(), status='READY')
        assert final['Architecture'] == 'arm64'
        write(self.root / 'ready.json', receipt)
        print(json.dumps(receipt, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('platform', choices=JOBS)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--verify-existing', action='store_true', help='Verify the already-built target without a new build')
    args = p.parse_args()
    if not args.execute:
        print(json.dumps({'platform': args.platform, 'job': JOBS[args.platform], 'mode': 'PLAN_ONLY'}))
        return
    assert (os.getuid(), os.getgid()) == (28644, 30)
    image = Image(args.platform)
    try:
        image.run(args.verify_existing)
    except BaseException as exc:
        write(image.root / 'error.json', {'at': utc(), 'type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
