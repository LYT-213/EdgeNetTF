#!/usr/bin/env python3
"""FFT-window sensitivity analysis for EdgeNetTF on NinaPro DB2.

Compares Hann and Hamming tapering against the original direct FFT (rectangular
equivalent). Only the taper changes; all other training and evaluation settings
match the primary protocol. Edit BASE_PATH before running.
"""

import gc, os, random
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from src.edgenettf.models import EdgeNetTF
from src.edgenettf.preprocessing import compress_semg
from src.edgenettf.smoothing import segmented_causal_majority_vote

BASE_PATH="/kaggle/input/datasets/quddusikashaf/ninapro-db2"
TRAIN_REPS=[1,3,4,6]; TEST_REPS=[2,5]; WINDOW_SIZE=600; TRAIN_STRIDE=60; TEST_STRIDE=600
EPOCHS=60; BATCH_SIZE=128; LR=1e-3; WEIGHT_DECAY=1e-3; MIXUP_PROB=.3; MIXUP_ALPHA=.2; LABEL_SMOOTHING=.05
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOWS={'hann':np.hanning(WINDOW_SIZE).astype(np.float32),'hamming':np.hamming(WINDOW_SIZE).astype(np.float32)}


def seed_all(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def load_subject(sid):
 d=os.path.join(BASE_PATH,f'DB2_s{sid}'); paths=[]
 for root,_,files in os.walk(d): paths += [os.path.join(root,f) for f in files if f.endswith('.mat')]
 E=[];L=[];R=[]
 for p in sorted(paths):
  m=sio.loadmat(p); E.append(compress_semg(m['emg'])); L.append(m['restimulus']); R.append(m['rerepetition'])
 return np.vstack(E),np.vstack(L).reshape(-1),np.vstack(R).reshape(-1)


def windows(emg,labels,reps,target_reps,stride,taper):
 xt=[];xf=[];yy=[];ss=[]; seg_id=0
 for rep in target_reps:
  idx=np.where(reps==rep)[0]
  for block in np.split(idx,np.where(np.diff(idx)>1)[0]+1):
   if len(block)<WINDOW_SIZE: continue
   bl=labels[block]; cps=np.where(np.diff(bl)!=0)[0]+1
   for seg in np.split(block,cps):
    if len(seg)<WINDOW_SIZE: continue
    label=int(labels[seg[0]]); cur=seg_id; seg_id+=1
    if not 1<=label<=49: continue
    for i in range(0,len(seg)-WINDOW_SIZE+1,stride):
     w=emg[seg[i:i+WINDOW_SIZE]]; spec=np.fft.fft(w*taper[:,None],axis=0)[:WINDOW_SIZE//2]
     xt.append(w.astype(np.float32)); xf.append(np.log1p(np.abs(spec)+1e-8).astype(np.float32)); yy.append(label-1); ss.append(cur)
 return np.asarray(xt),np.asarray(xf),np.asarray(yy),np.asarray(ss)


def standardize(a,b):
 mean=a.mean(axis=(0,1),keepdims=True); sd=a.std(axis=(0,1),keepdims=True)+1e-8; return (a-mean)/sd,(b-mean)/sd

def ten(x): return torch.tensor(x,dtype=torch.float32).permute(0,2,1)


def run(sid,name,taper):
 seed_all(2000+sid); emg,lab,rep=load_subject(sid); a,b,y,_=windows(emg,lab,rep,TRAIN_REPS,TRAIN_STRIDE,taper); c,d,z,seg=windows(emg,lab,rep,TEST_REPS,TEST_STRIDE,taper)
 a,c=standardize(a,c); b,d=standardize(b,d)
 tr=TensorDataset(ten(a),ten(b),torch.tensor(y,dtype=torch.long)); te=TensorDataset(ten(c),ten(d),torch.tensor(z,dtype=torch.long),torch.tensor(seg,dtype=torch.long))
 tl=DataLoader(tr,batch_size=BATCH_SIZE,shuffle=True,num_workers=2); vl=DataLoader(te,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)
 model=EdgeNetTF().to(DEVICE); opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY); sched=optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=20,T_mult=2); ce=nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
 for _ in range(EPOCHS):
  model.train()
  for xt,xf,yy in tl:
   xt=xt.to(DEVICE); xf=xf.to(DEVICE); yy=yy.to(DEVICE); opt.zero_grad()
   if random.random()<MIXUP_PROB:
    lam=np.random.beta(MIXUP_ALPHA,MIXUP_ALPHA); idx=torch.randperm(yy.size(0),device=DEVICE); out=model(lam*xt+(1-lam)*xt[idx],lam*xf+(1-lam)*xf[idx]); loss=lam*ce(out,yy)+(1-lam)*ce(out,yy[idx])
   else: loss=ce(model(xt,xf),yy)
   loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
  sched.step()
 model.eval(); P=[];Y=[];S=[]
 with torch.no_grad():
  for xt,xf,yy,ss in vl: P.append(model(xt.to(DEVICE),xf.to(DEVICE)).argmax(1).cpu().numpy()); Y.append(yy.numpy()); S.append(ss.numpy())
 p=np.concatenate(P); yy=np.concatenate(Y); ss=np.concatenate(S); q=segmented_causal_majority_vote(p,ss,5)
 row={'FFT_Window':name,'Subject':sid,'Raw_Acc':accuracy_score(yy,p),'Raw_F1_Macro':f1_score(yy,p,average='macro',zero_division=0),'Causal5_Acc':accuracy_score(yy,q),'Causal5_F1_Macro':f1_score(yy,q,average='macro',zero_division=0)}
 del model,tr,te,tl,vl; gc.collect(); torch.cuda.empty_cache(); return row


def main():
 out=Path('results/revision/fft_window_subject_level.csv'); out.parent.mkdir(parents=True,exist_ok=True); rows=[]
 for name,taper in WINDOWS.items():
  for sid in range(1,41):
   print(name,sid,flush=True); rows.append(run(sid,name,taper)); pd.DataFrame(rows).to_csv(out,index=False)
 df=pd.DataFrame(rows); print(df.groupby('FFT_Window')[['Raw_Acc','Raw_F1_Macro','Causal5_Acc','Causal5_F1_Macro']].agg(['mean',lambda x:x.std(ddof=1)]))

if __name__=='__main__': main()
