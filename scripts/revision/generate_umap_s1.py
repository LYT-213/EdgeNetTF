#!/usr/bin/env python3
"""Generate the revision UMAP visualization for representative DB2 subject S1.

The script trains the original EdgeNetTF for S1 with the primary repetition
split and extracts temporal-branch, frequency-branch, and concatenated features
from held-out repetitions. Each feature space is independently standardized and
projected with UMAP (n_neighbors=30, min_dist=0.1, Euclidean, random_state=42).
Edit BASE_PATH before running.
"""

import os, random
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import matplotlib.pyplot as plt
from src.edgenettf.models import EdgeNetTF
from src.edgenettf.preprocessing import compress_semg, create_segmented_windows, standardize_train_test

BASE_PATH='/kaggle/input/datasets/quddusikashaf/ninapro-db2'; SUBJECT=1
TRAIN_REPS=[1,3,4,6]; TEST_REPS=[2,5]; WINDOW_SIZE=600; TRAIN_STRIDE=60; TEST_STRIDE=600
EPOCHS=60; BATCH_SIZE=128; DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_all(seed=42):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def load():
 d=os.path.join(BASE_PATH,f'DB2_s{SUBJECT}'); paths=[]
 for root,_,files in os.walk(d): paths += [os.path.join(root,f) for f in files if f.endswith('.mat')]
 E=[];L=[];R=[]
 for p in sorted(paths):
  m=sio.loadmat(p); E.append(compress_semg(m['emg'])); L.append(m['restimulus']); R.append(m['rerepetition'])
 return np.vstack(E),np.vstack(L),np.vstack(R)

def ten(x): return torch.tensor(x,dtype=torch.float32).permute(0,2,1)


def main():
 seed_all(42); emg,lab,rep=load()
 a,b,y,_=create_segmented_windows(emg,lab,rep,TRAIN_REPS,WINDOW_SIZE,TRAIN_STRIDE,1,49)
 c,d,z,_=create_segmented_windows(emg,lab,rep,TEST_REPS,WINDOW_SIZE,TEST_STRIDE,1,49)
 a,c=standardize_train_test(a,c); b,d=standardize_train_test(b,d); y=y-1; z=z-1
 tr=TensorDataset(ten(a),ten(b),torch.tensor(y,dtype=torch.long)); tl=DataLoader(tr,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
 model=EdgeNetTF().to(DEVICE); opt=optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3); sched=optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=20,T_mult=2); ce=nn.CrossEntropyLoss(label_smoothing=.05)
 for _ in range(EPOCHS):
  model.train()
  for xt,xf,yy in tl:
   xt=xt.to(DEVICE); xf=xf.to(DEVICE); yy=yy.to(DEVICE); opt.zero_grad()
   if random.random()<.3:
    lam=np.random.beta(.2,.2); idx=torch.randperm(yy.size(0),device=DEVICE); out=model(lam*xt+(1-lam)*xt[idx],lam*xf+(1-lam)*xf[idx]); loss=lam*ce(out,yy)+(1-lam)*ce(out,yy[idx])
   else: loss=ce(model(xt,xf),yy)
   loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
  sched.step()
 model.eval()
 with torch.no_grad():
  xt=ten(c).to(DEVICE); xf=ten(d).to(DEVICE); tf=model.time_stem(xt).cpu().numpy(); ff=model.freq_stem(xf).cpu().numpy(); fused=np.concatenate([tf,ff],axis=1)
 spaces=[('Temporal features',tf),('Frequency-domain features',ff),('Fused features',fused)]
 fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
 for ax,(title,feat),label in zip(axes,spaces,['(a)','(b)','(c)']):
  feat=StandardScaler().fit_transform(feat); emb=umap.UMAP(n_neighbors=30,min_dist=.1,metric='euclidean',random_state=42).fit_transform(feat)
  sc=ax.scatter(emb[:,0],emb[:,1],c=z+1,s=6,cmap='turbo',alpha=.8); ax.set_title(f'{label} {title}'); ax.set_xticks([]); ax.set_yticks([])
 cbar=fig.colorbar(sc,ax=axes.tolist(),fraction=.025,pad=.02); cbar.set_label('Gesture class'); plt.savefig('EdgeNetTF_UMAP_S1.png',dpi=300,bbox_inches='tight')

if __name__=='__main__': main()
