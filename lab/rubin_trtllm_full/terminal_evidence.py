"""Offline validation of the campaign's retained terminal Ray observation.

No remote calls or mutations. A completed curve is never terminal evidence.
The profile operator retains these records before stopping the main container.
"""
import datetime as dt
import hashlib
import json
import re
import shlex


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def profile_root(c):
    if not re.fullmatch(r'20260917-' + re.escape(c['platform']) + r'-j' + re.escape(str(c['job_id'])) + r'-trtllm', c['run_id']):
        return None
    return f"diagnostics/{c['platform']}-j{c['job_id']}-trtllm-profile-v1"


def artifact_paths(c):
    root = profile_root(c)
    if root is None:
        return ()
    return ('campaign-main-completion.json', 'cudagraph-main-plan.json', *(
        root + '/' + name for name in (
            'operator-plan.json', 'main-driver-evidence.json', 'main-login-retention.json',
            'main-retention.json', 'main-before/watchdog.json',
            'main-final-retention.json', 'main-final/watchdog.json')))


def validate(c, files):
    """Return verified watchdog/Ray/receipt or None while retention is incomplete."""
    root = profile_root(c)
    if root is None:
        return None
    required = ['campaign-main-completion.json', 'cudagraph-main-plan.json',
                'driver-config.json', 'source-manifest.json', 'train-launch.json',
                'train_exit.json', 'train-driver-exit.json', 'logs/qwen3_train.log',
                root + '/operator-plan.json', root + '/main-driver-evidence.json',
                root + '/main-login-retention.json', root + '/main-retention.json',
                root + '/main-before/watchdog.json']
    if any(name not in files for name in required):
        return None
    used = set(required)

    def require(condition, reason):
        if not condition:
            raise ValueError('Retained terminal evidence: ' + reason)

    def read(name):
        used.add(name)
        return json.loads(files[name])

    def verify_raw(name, record):
        require(isinstance(record, dict) and record.get('sha256') == digest(files[name])
                and record.get('bytes') == len(files[name]), 'raw SHA/size mismatch: ' + name)

    operator = read(root + '/operator-plan.json')
    original = read('cudagraph-main-plan.json')
    drivers = read(root + '/main-driver-evidence.json')
    launch = read('train-launch.json')
    source = read('source-manifest.json')
    complete = read('campaign-main-completion.json')
    login = read(root + '/main-login-retention.json')
    before = read(root + '/main-retention.json')
    require(read('driver-config.json') == c, 'driver configuration differs')
    for label, value in [('operator', operator.get('config', {})), ('main plan', original.get('config', {}))]:
        require(all(value.get(k) == v for k, v in c.items()), label + ' configuration differs')
    require(original.get('run_id') == c['run_id'] and complete.get('run_id') == c['run_id'], 'run identity differs')
    require(re.fullmatch(r'[0-9a-f]{40}', c['source_commit']) is not None, 'invalid source commit')
    source_sha = digest(files['source-manifest.json'])
    for label, value in [('source manifest', source), ('main plan', original), ('launch', launch)]:
        require(value.get('git_commit') == c['source_commit'], label + ' source commit differs')
    require(original.get('source_manifest_sha256') == launch.get('source_manifest_sha256') == source_sha,
            'source manifest hash differs')
    require(original.get('source_sha256') and original['source_sha256'] == launch.get('source_sha256')
            and all(source.get('source_sha256', {}).get(k) == v for k, v in original['source_sha256'].items()),
            'launched source hashes differ')
    require(drivers.get('original_main_plan') == original, 'retained main plan differs')
    for key, name in [('driver_exit', 'train-driver-exit.json'), ('train_exit', 'train_exit.json')]:
        value = read(name)
        require(value.get('exit_code') == 0 and drivers.get(key) == value, 'driver/train exit differs or is nonzero')
        require(value.get('run_id', c['run_id']) == c['run_id'], 'driver/train run differs')
    require(read('train-driver-exit.json').get('run_id') == c['run_id'], 'driver exit lacks exact run identity')
    require(login.get('stable_source_and_destination_sha256_verified') is True
            and before.get('source_prefix_and_destination_sha256_verified') is True,
            'retention verification was not completed')
    for name in ('logs/qwen3_train.log', 'train-launch.json', 'cudagraph-main-plan.json',
                 'train_exit.json', 'train-driver-exit.json', 'source-manifest.json'):
        record = login.get('files', {}).get(name)
        verify_raw(name, record)
        require(before.get('login_files', {}).get(name) == record, 'login retention receipts disagree: ' + name)
    require(complete.get('source_log_sha256') == digest(files['logs/qwen3_train.log']), 'completion log hash differs')
    require(complete.get('completion_validation', {}).get('status') == 'PASS', 'campaign completion audit did not pass')

    container = operator.get('main_container', {})
    preflight = original.get('preflight', {})
    require(container.get('Name') == '/' + c['container_prefix'] + '-0'
            and re.fullmatch(r'[0-9a-f]{64}', container.get('Id', '')) is not None
            and preflight.get('container_id') == container['Id'], 'main container identity differs')
    require(container.get('Image') == operator['config'].get('image_id') == preflight.get('image_id')
            and re.fullmatch(r'sha256:[0-9a-f]{64}', container.get('Image', '')) is not None,
            'inspected main image ID differs')

    watch_path, watch_receipt = root + '/main-before/watchdog.json', root + '/main-retention.json'
    verify_raw(watch_path, before.get('files', {}).get('watchdog.json'))
    before_watch = read(watch_path)
    final_path, final_receipt = root + '/main-final/watchdog.json', root + '/main-final-retention.json'
    if final_path in files and final_receipt in files:
        verify_raw(final_path, read(final_receipt).get('watchdog.json'))
        require(read(final_path) == before_watch, 'terminal watchdog changed after main stop')
        watch_path, watch_receipt = final_path, final_receipt
    watchdog = read(watch_path)
    require(watchdog == complete.get('watchdog'), 'campaign and retained watchdog disagree')
    require(watchdog.get('run_id') == c['run_id'] and watchdog.get('state') == 'finished'
            and watchdog.get('terminal_status') == 'SUCCEEDED' and watchdog.get('stop_request_count') == 0
            and watchdog.get('expected_rollouts') == 50, 'watchdog does not prove clean terminal success')
    lease = dt.datetime.fromisoformat(c['lease_deadline'].replace('Z', '+00:00'))
    for key, minutes in [('lease_deadline_at', 0), ('soft_deadline_at', 90), ('deadline_at', 60)]:
        require(dt.datetime.fromisoformat(watchdog.get(key, '').replace('Z', '+00:00')) == lease - dt.timedelta(minutes=minutes),
                'watchdog original deadline differs: ' + key)

    ray = operator.get('ray_job', {})
    env = ray.get('runtime_env') or {}
    if isinstance(env, str):
        env = json.loads(env)
    require((env.get('env_vars') or {}).get('RUBIN_RUN_ID') == c['run_id'], 'Ray runtime run identity differs')
    require(ray.get('status') == 'SUCCEEDED' and ray.get('driver_exit_code') in (None, 0)
            and isinstance(ray.get('submission_id'), str) and ray['submission_id']
            and ray['submission_id'] == watchdog.get('submission_id'), 'actual Ray job is not this successful submission')
    require(all(complete.get('ray_job', {}).get(k) == ray.get(k) for k in
                ('submission_id', 'status', 'driver_exit_code', 'start_time', 'end_time')), 'Ray terminal receipts disagree')
    entries = re.findall(r'^Running entrypoint for job (\S+): (.+)$', files['logs/qwen3_train.log'].decode(errors='replace'), re.M)
    require(len(entries) == 1 and entries[0][0] == ray['submission_id']
            and isinstance(ray.get('entrypoint'), str)
            and shlex.split(entries[0][1]) == shlex.split(ray['entrypoint']), 'actual Ray/logged entrypoint differs')
    normalized = {k: ray.get(k) for k in ('status', 'entrypoint', 'submission_id', 'job_id', 'driver_exit_code', 'start_time', 'end_time')}
    normalized.update(run_id=c['run_id'], observed_at=complete['at'])
    receipt = {'schema': 'trtllm-retained-terminal-import-v1', 'status': 'VERIFIED', 'run_id': c['run_id'],
               'submission_id': ray['submission_id'], 'ray_status': ray['status'], 'observed_at': complete['at'],
               'watchdog_path': watch_path, 'watchdog_receipt': watch_receipt,
               'ray_path': root + '/operator-plan.json', 'image': c['image'], 'image_id': container['Image'],
               'source_commit': c['source_commit'], 'source_log_sha256': digest(files['logs/qwen3_train.log']),
               'source_artifacts': {name: {'bytes': len(files[name]), 'sha256': digest(files[name])} for name in sorted(used)},
               'scope': 'Actual retained Ray SUCCEEDED observation and watchdog, bound to original image/source/submission/log. Full 50/200/800 validation is recomputed by the collector.'}
    return {'watchdog': watchdog, 'ray': normalized, 'receipt': receipt}
