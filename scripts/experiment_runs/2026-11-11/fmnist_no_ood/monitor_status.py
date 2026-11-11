#!/usr/bin/env python3
from __future__ import annotations
import os, re, time, subprocess
from pathlib import Path
from datetime import datetime, timezone

base = Path('/ssd/haibin/logs/topology_aware_learning_fmnist_no_noise_20261111')
queue = base / 'queue_gpucache_resume.log'
now = datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')
active = []
for p in Path('/proc').iterdir():
    if not p.name.isdigit():
        continue
    try:
        cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        if 'src.experiments.decentralized_main' not in cmd or not '_2xgpu7' in cmd:
            continue
        env = dict(x.split('=', 1) for x in (p / 'environ').read_bytes().decode(errors='ignore').split('\0') if '=' in x)
        match = re.search(r'--run_id\s+(\S+)', cmd)
        run_id = match.group(1) if match else 'unknown'
        gpu = env.get('CUDA_VISIBLE_DEVICES', '?')
        logs = list((base / 'results' / run_id).glob('*/experiment.log'))
        round_idx = None
        mtime = None
        if logs:
            f = max(logs, key=lambda x: x.stat().st_mtime)
            mtime = f.stat().st_mtime
            try:
                data = f.read_text(errors='replace')
                rounds = re.findall(r'\((\d+)/40\) Starting local training', data)
                round_idx = int(rounds[-1]) if rounds else 0
            except OSError:
                pass
        active.append((int(p.name), gpu, run_id, round_idx, mtime))
    except (OSError, PermissionError):
        continue
queue_text = queue.read_text(errors='replace') if queue.exists() else ''
starts = len(re.findall(r'\bSTART\b', queue_text))
dones = len(re.findall(r'\bDONE\b', queue_text))
fails = len(re.findall(r'\bFAIL\b', queue_text))
all_done = 'ALL_DONE' in queue_text
try:
    gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader'], text=True, timeout=10).strip().replace('\n', '; ')
except Exception as e:
    gpus = f'query_error={type(e).__name__}:{e}'
notes = []
if not all_done and len(active) < 14:
    notes.append(f'active_processes={len(active)}/14')
if fails:
    notes.append(f'failed_runs={fails}')
for pid, gpu, run_id, rnd, mtime in active:
    if mtime is not None and time.time() - mtime > 3600:
        notes.append(f'stale_log>{int((time.time()-mtime)/60)}min:{run_id}')
health = 'WARN ' + ','.join(notes) if notes else 'OK'
print(f'{now} {health} queue=START:{starts},DONE:{dones},FAIL:{fails},ALL_DONE:{all_done} active={len(active)}')
for pid, gpu, run_id, rnd, mtime in sorted(active, key=lambda x: x[1]):
    age = 'n/a' if mtime is None else f'{int((time.time()-mtime)/60)}m'
    print(f'  gpu={gpu} pid={pid} run={run_id} last_round={rnd if rnd is not None else "?"}/40 log_age={age}')
print(f'  nvidia_smi={gpus}')
