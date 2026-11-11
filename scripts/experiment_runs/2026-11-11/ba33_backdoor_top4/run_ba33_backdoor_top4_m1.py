#!/usr/bin/env python3
from __future__ import annotations
import os, subprocess, time
from pathlib import Path
ROOT=Path('/home/haibin/topo_fl/topology_aware_learning')
BASE=Path('/ssd/haibin/logs/ba33_backdoor_top4_m1_20261111')
PYTHON='/ssd/haibin/miniconda3/envs/topo/bin/python'
DATA='/ssd/haibin/datasets'
GPUS=[0,3,4,5,6,7]
AGGS=['unweighted','random','random_betCent','random_degCent','betCent','degCent']
OLD_RUNNERS=['run_cifar10_noise_20.sh','retry_failed_now.sh','run_moved_pending_gpu7.sh']
BASE.mkdir(parents=True,exist_ok=True)
for s in ('results','stdout','parsl'): (BASE/s).mkdir(exist_ok=True)
def log(msg):
 line=f'{time.strftime("%Y-%m-%dT%H:%M:%S%z")} {msg}'; print(line,flush=True)
 with (BASE/'scheduler.log').open('a') as f: f.write(line+'\n')
def launch(gpu,agg):
 rid=f'cifar10_ba33_m1_badnets_nodes0-2-5-8_{agg}_s1_finalonly'
 env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':str(gpu),'TOPO_FL_LOCAL_THREADS':'8','TOPO_FL_FINAL_EVAL_ONLY':'1','TOPO_FL_INPROCESS':'1','TOPO_FL_PIN_MEMORY':'0','TOPO_FL_VALIDATE_TARGET_TRANSFER':'1','TOPO_FL_PARSL_RUN_DIR':str(BASE/'parsl'/f'parsl_runinfo_{rid}'),'TOPO_FL_PARSL_LOG':str(BASE/'parsl'/f'parsl_{rid}.log'),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'})
 out=(BASE/'stdout'/f'{rid}.stdout.log').open('w'); err=(BASE/'stdout'/f'{rid}.stderr.log').open('w')
 cmd=[PYTHON,'-m','src.experiments.decentralized_main','--rounds','40','--checkpoint_every','5','--epochs','5','--batch_size','16','--lr','0.0001','--optimizer','adam','--weight_decay','0','--participation','1.0','--sample_alpha','1000','--label_alpha','1000','--aggregation_strategy',agg,'--dataset','cifar10_vgg','--topology_file','src/create_topo/topology/topo_ba33.txt','--backdoor','--backdoor_proportion','0.1','--backdoor_node_idxs','0','2','5','8','--out_dir',str(BASE/'results'),'--data_dir',DATA,'--run_id',rid,'--seed','1','--parsl_executor','local']
 p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=err); p._files=(out,err); log(f'START gpu={gpu} pid={p.pid} agg={agg} run={rid}'); return p
def old_runners():
 try: lines=subprocess.check_output(['ps','-eo','args='],text=True).splitlines()
 except subprocess.CalledProcessError: return []
 return [x for x in lines if any(n in x for n in OLD_RUNNERS) and 'run_ba33_backdoor_top4_m1.py' not in x]
# Idle GPU 0 is safe to use now; run two tasks there, within the established per-GPU cap.
first=[launch(0,a) for a in AGGS[:2]]
for p in first:
 c=p.wait(); [f.close() for f in p._files]; log(f'EXIT pid={p.pid} code={c}')
# The current Gaussian-noise matrix still owns GPUs 3-7; avoid racing its queue workers.
while old_runners():
 log('WAIT existing CIFAR-noise queue/retry runners to finish before using their GPUs'); time.sleep(60)
# Run the remaining configurations after the prior runners exit.
running={}; i=0
while i<len(AGGS[2:]) or running:
 while i<len(AGGS[2:]) and len(running)<len(GPUS):
  agg=AGGS[2+i]; gpu=GPUS[i%len(GPUS)]; p=launch(gpu,agg); running[p]=(gpu,agg); i+=1
 for p,(gpu,agg) in list(running.items()):
  c=p.poll()
  if c is not None:
   [f.close() for f in p._files]; log(f'EXIT gpu={gpu} agg={agg} pid={p.pid} code={c}'); del running[p]
 if running: time.sleep(5)
log('ALL_DONE')
