#!/usr/bin/env python3
"""End-to-end CPU latency benchmark used in the major revision.

Measures logarithmic compression + FFT + normalization + tensor conversion +
EdgeNetTF inference for one 300-ms window. The model weights do not affect the
operation count, so a checkpoint is not required for latency measurement.
Edit BASE_PATH before running.
"""

import os, time
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from src.edgenettf.models import EdgeNetTF
from src.edgenettf.preprocessing import compress_semg, create_segmented_windows

BASE_PATH='/kaggle/input/datasets/quddusikashaf/ninapro-db2'; SUBJECT=1
TRAIN_REPS=[1,3,4,6]; TEST_REPS=[2,5]; WINDOW_SIZE=600


def load():
 d=os.path.join(BASE_PATH,f'DB2_s{SUBJECT}'); paths=[]
 for root,_,files in os.walk(d): paths += [os.path.join(root,f) for f in files if f.endswith('.mat')]
 E=[];L=[];R=[]
 for p in sorted(paths):
  m=sio.loadmat(p); E.append(m['emg']); L.append(m['restimulus']); R.append(m['rerepetition'])
 return np.vstack(E),np.vstack(L),np.vstack(R)


def main():
 torch.set_num_threads(1); raw,lab,rep=load(); compressed=compress_semg(raw)
 tr_t,tr_f,_,_=create_segmented_windows(compressed,lab,rep,TRAIN_REPS,600,60,1,49)
 t_mean=tr_t.mean(axis=(0,1),keepdims=True); t_std=tr_t.std(axis=(0,1),keepdims=True)+1e-8
 f_mean=tr_f.mean(axis=(0,1),keepdims=True); f_std=tr_f.std(axis=(0,1),keepdims=True)+1e-8
 raw_window=None
 for r in TEST_REPS:
  idx=np.where(rep.reshape(-1)==r)[0]
  for block in np.split(idx,np.where(np.diff(idx)>1)[0]+1):
   bl=lab[block,0].astype(int); cps=np.where(np.diff(bl)!=0)[0]+1
   for seg in np.split(block,cps):
    if len(seg)>=600 and 1<=int(lab[seg[0],0])<=49: raw_window=raw[seg[:600]].astype(np.float32); break
   if raw_window is not None: break
  if raw_window is not None: break
 model=EdgeNetTF().cpu().eval()
 def once(rw):
  x=compress_semg(rw); spec=np.fft.fft(x,axis=0)[:300]; xf=np.log1p(np.abs(spec)+1e-8)
  xt=(x[None]-t_mean)/t_std; xf=(xf[None]-f_mean)/f_std
  xt=torch.tensor(xt,dtype=torch.float32).permute(0,2,1); xf=torch.tensor(xf,dtype=torch.float32).permute(0,2,1)
  with torch.no_grad(): return model(xt,xf)
 for _ in range(100): once(raw_window)
 lat=[]
 for _ in range(1000):
  s=time.perf_counter(); once(raw_window); lat.append((time.perf_counter()-s)*1000)
 lat=np.asarray(lat)
 summary={'Mean_ms':lat.mean(),'Sample_SD_ms':lat.std(ddof=1),'Median_ms':np.median(lat),'P95_ms':np.percentile(lat,95),'Warmup_Runs':100,'Timed_Runs':1000}
 print(summary); pd.DataFrame({'Latency_ms':lat}).to_csv('results/revision/end_to_end_latency_runs.csv',index=False); pd.DataFrame([summary]).to_csv('results/revision/end_to_end_latency_summary.csv',index=False)

if __name__=='__main__': main()
