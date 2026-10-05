#!/usr/bin/env python3
"""Kernel-size and channel-width sensitivity analysis for EdgeNetTF.

Runs only the four revision variants; the original EdgeNetTF results are taken
from the primary experiment. Protocol otherwise matches the primary DB2 setup.
Edit BASE_PATH before running.
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
from src.edgenettf.preprocessing import compress_semg, create_segmented_windows
from src.edgenettf.smoothing import segmented_causal_majority_vote

BASE_PATH="/kaggle/input/datasets/quddusikashaf/ninapro-db2"
TRAIN_REPS=[1,3,4,6]; TEST_REPS=[2,5]; WINDOW_SIZE=600; TRAIN_STRIDE=60; TEST_STRIDE=600
EPOCHS=60; BATCH_SIZE=128; LR=1e-3; WEIGHT_DECAY=1e-3; MIXUP_PROB=0.3; MIXUP_ALPHA=0.2; LABEL_SMOOTHING=0.05
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VARIANTS={
 'kernel_small':dict(t1=3,t2=3,f1=3,f2=3,c1=64,c2=128),
 'kernel_large':dict(t1=9,t2=9,f1=9,f2=7,c1=64,c2=128),
 'channel_narrow':dict(t1=7,t2=7,f1=7,f2=5,c1=32,c2=64),
 'channel_wide':dict(t1=7,t2=7,f1=7,f2=5,c1=128,c2=256),
}


def seed_all(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def load_subject(sid):
 d=os.path.join(BASE_PATH,f'DB2_s{sid}'); paths=[]
 for root,_,files in os.walk(d): paths += [os.path.join(root,f) for f in files if f.endswith('.mat')]
 E=[]; L=[]; R=[]
 for p in sorted(paths):
  m=sio.loadmat(p); E.append(compress_semg(m['emg'])); L.append(m['restimulus']); R.append(m['rerepetition'])
 return np.vstack(E),np.vstack(L),np.vstack(R)


def standardize(a,b):
 mean=a.mean(axis=(0,1),keepdims=True); sd=a.std(axis=(0,1),keepdims=True)+1e-8; return (a-mean)/sd,(b-mean)/sd


def ten(x): return torch.tensor(x,dtype=torch.float32).permute(0,2,1)


class VariantNet(nn.Module):
 def __init__(self,cfg):
  super().__init__(); c1,c2=cfg['c1'],cfg['c2']
  self.t=nn.Sequential(nn.Conv1d(12,c1,cfg['t1'],padding=cfg['t1']//2),nn.BatchNorm1d(c1),nn.GELU(),nn.MaxPool1d(2),nn.Conv1d(c1,c2,cfg['t2'],padding=cfg['t2']//2),nn.BatchNorm1d(c2),nn.GELU(),nn.MaxPool1d(2),nn.AdaptiveAvgPool1d(1))
  self.f=nn.Sequential(nn.Conv1d(12,c1,cfg['f1'],padding=cfg['f1']//2),nn.BatchNorm1d(c1),nn.GELU(),nn.Conv1d(c1,c2,cfg['f2'],padding=cfg['f2']//2),nn.BatchNorm1d(c2),nn.GELU(),nn.AdaptiveAvgPool1d(1))
  self.cls=nn.Sequential(nn.Linear(c2*2,256),nn.LayerNorm(256),nn.GELU(),nn.Dropout(.3),nn.Linear(256,49))
 def forward(self,xt,xf): return self.cls(torch.cat([self.t(xt).squeeze(-1),self.f(xf).squeeze(-1)],1))


def run(sid,name,cfg):
 seed_all(2000+sid); emg,lab,rep=load_subject(sid)
 a,b,y,_=create_segmented_windows(emg,lab,rep,TRAIN_REPS,WINDOW_SIZE,TRAIN_STRIDE,1,49)
 c,d,z,seg=create_segmented_windows(emg,lab,rep,TEST_REPS,WINDOW_SIZE,TEST_STRIDE,1,49)
 y=y-1; z=z-1; a,c=standardize(a,c); b,d=standardize(b,d)
 tr=TensorDataset(ten(a),ten(b),torch.tensor(y,dtype=torch.long)); te=TensorDataset(ten(c),ten(d),torch.tensor(z,dtype=torch.long),torch.tensor(seg,dtype=torch.long))
 tl=DataLoader(tr,batch_size=BATCH_SIZE,shuffle=True,num_workers=2); vl=DataLoader(te,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)
 model=VariantNet(cfg).to(DEVICE); opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY); sched=optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=20,T_mult=2); ce=nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
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
 row={'Variant':name,'Subject':sid,'Params_M':sum(x.numel() for x in model.parameters() if x.requires_grad)/1e6,'Raw_Acc':accuracy_score(yy,p),'Raw_F1_Macro':f1_score(yy,p,average='macro',zero_division=0),'Causal5_Acc':accuracy_score(yy,q),'Causal5_F1_Macro':f1_score(yy,q,average='macro',zero_division=0)}
 del model,tr,te,tl,vl; gc.collect(); torch.cuda.empty_cache(); return row


def main():
 out=Path('results/revision/hyperparameter_subject_level.csv'); out.parent.mkdir(parents=True,exist_ok=True); rows=[]
 for name,cfg in VARIANTS.items():
  for sid in range(1,41):
   print(name,sid,flush=True); rows.append(run(sid,name,cfg)); pd.DataFrame(rows).to_csv(out,index=False)
 df=pd.DataFrame(rows); print(df.groupby('Variant')[['Raw_Acc','Raw_F1_Macro','Causal5_Acc','Causal5_F1_Macro']].agg(['mean',lambda x:x.std(ddof=1)]))

if __name__=='__main__': main()
