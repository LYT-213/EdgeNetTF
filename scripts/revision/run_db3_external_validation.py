#!/usr/bin/env python3
"""NinaPro DB3 external validation used in the Scientific Reports revision.

Per participant: repetitions 1/3/4/6 for training and 2/5 for testing,
600-sample windows at 2 kHz, train stride 60, test stride 600, training-only
normalization, original EdgeNetTF architecture, 60 epochs. Available active
classes are remapped to contiguous indices for each participant.

Edit BASE_PATH for your DB3 location before running.
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
from src.edgenettf.preprocessing import compress_semg, create_segmented_windows
from src.edgenettf.smoothing import segmented_causal_majority_vote

BASE_PATH="/kaggle/input/datasets/abdallaalaa/ninapro-dd3"
TRAIN_REPS=[1,3,4,6]; TEST_REPS=[2,5]
WINDOW_SIZE=600; TRAIN_STRIDE=60; TEST_STRIDE=600
EPOCHS=60; BATCH_SIZE=128; LR=1e-3; WEIGHT_DECAY=1e-3
MIXUP_PROB=0.3; MIXUP_ALPHA=0.2; LABEL_SMOOTHING=0.05
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def load_subject(sid):
    d=os.path.join(BASE_PATH,f"DB3_s{sid}")
    if not os.path.exists(d): d=os.path.join(BASE_PATH,f"s{sid}")
    paths=[]
    for root,_,files in os.walk(d): paths += [os.path.join(root,f) for f in files if f.endswith('.mat')]
    if not paths: raise FileNotFoundError(d)
    E=[]; L=[]; R=[]
    for p in sorted(paths):
        m=sio.loadmat(p); E.append(compress_semg(m['emg']))
        L.append(m['restimulus'] if 'restimulus' in m else m['stimulus'])
        R.append(m['rerepetition'] if 'rerepetition' in m else m['repetition'])
    return np.vstack(E),np.vstack(L),np.vstack(R)


def std_train_test(a,b):
    mean=a.mean(axis=(0,1),keepdims=True); sd=a.std(axis=(0,1),keepdims=True)+1e-8
    return (a-mean)/sd,(b-mean)/sd


def tensorize(x): return torch.tensor(x,dtype=torch.float32).permute(0,2,1)


def remap(ytr,yte):
    classes=np.unique(np.concatenate([ytr,yte])); mp={c:i for i,c in enumerate(classes)}
    return np.array([mp[x] for x in ytr]),np.array([mp[x] for x in yte]),len(classes)


def run_subject(sid):
    seed_all(4000+sid); emg,lab,rep=load_subject(sid)
    xtr_t,xtr_f,ytr,_=create_segmented_windows(emg,lab,rep,TRAIN_REPS,WINDOW_SIZE,TRAIN_STRIDE,1,99)
    xte_t,xte_f,yte,seg=create_segmented_windows(emg,lab,rep,TEST_REPS,WINDOW_SIZE,TEST_STRIDE,1,99)
    ytr,yte,ncls=remap(ytr,yte)
    xtr_t,xte_t=std_train_test(xtr_t,xte_t); xtr_f,xte_f=std_train_test(xtr_f,xte_f)
    tr=TensorDataset(tensorize(xtr_t),tensorize(xtr_f),torch.tensor(ytr,dtype=torch.long))
    te=TensorDataset(tensorize(xte_t),tensorize(xte_f),torch.tensor(yte,dtype=torch.long),torch.tensor(seg,dtype=torch.long))
    tl=DataLoader(tr,batch_size=BATCH_SIZE,shuffle=True,num_workers=2); vl=DataLoader(te,batch_size=BATCH_SIZE,shuffle=False,num_workers=2)
    model=EdgeNetTF(num_classes=ncls).to(DEVICE); opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched=optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=20,T_mult=2); ce=nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    for _ in range(EPOCHS):
        model.train()
        for xt,xf,y in tl:
            xt=xt.to(DEVICE); xf=xf.to(DEVICE); y=y.to(DEVICE); opt.zero_grad()
            if random.random()<MIXUP_PROB:
                lam=np.random.beta(MIXUP_ALPHA,MIXUP_ALPHA); idx=torch.randperm(y.size(0),device=DEVICE)
                logits=model(lam*xt+(1-lam)*xt[idx],lam*xf+(1-lam)*xf[idx]); loss=lam*ce(logits,y)+(1-lam)*ce(logits,y[idx])
            else: loss=ce(model(xt,xf),y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        sched.step()
    model.eval(); P=[]; Y=[]; S=[]
    with torch.no_grad():
        for xt,xf,y,s in vl:
            P.append(model(xt.to(DEVICE),xf.to(DEVICE)).argmax(1).cpu().numpy()); Y.append(y.numpy()); S.append(s.numpy())
    p=np.concatenate(P); y=np.concatenate(Y); s=np.concatenate(S); smooth=segmented_causal_majority_vote(p,s,5)
    row={'Subject':sid,'Active_Classes':ncls,'Train_Windows':len(ytr),'Test_Windows':len(yte),'Raw_Acc':accuracy_score(y,p),'Raw_F1_Macro':f1_score(y,p,average='macro',zero_division=0),'Causal5_Acc':accuracy_score(y,smooth),'Causal5_F1_Macro':f1_score(y,smooth,average='macro',zero_division=0)}
    del model,tr,te,tl,vl; gc.collect(); torch.cuda.empty_cache(); return row


def main():
    out=Path('results/revision/db3_subject_level.csv'); out.parent.mkdir(parents=True,exist_ok=True); rows=[]
    for sid in range(1,12):
        print(f'DB3 S{sid}',flush=True); rows.append(run_subject(sid)); pd.DataFrame(rows).to_csv(out,index=False)
    df=pd.DataFrame(rows); print(df); print(df[['Raw_Acc','Raw_F1_Macro','Causal5_Acc','Causal5_F1_Macro']].agg(['mean',lambda x:x.std(ddof=1)]))

if __name__=='__main__': main()
