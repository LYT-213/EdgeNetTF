#!/usr/bin/env python3
"""Leakage-free LOSO evaluation on NinaPro DB2 used in the major revision.

Protocol: one held-out subject per fold; remaining 39 subjects for training;
all six repetitions; 600-sample (300 ms) non-overlapping windows for both
training and testing; normalization statistics estimated from training subjects
only; EdgeNetTF trained for 60 epochs.

Edit BASE_PATH for your local/Kaggle DB2 location before running.
"""

import argparse
import gc
import os
import random
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

BASE_PATH = "/kaggle/input/datasets/quddusikashaf/ninapro-db2"
SUBJECTS = list(range(1, 41))
ALL_REPS = [1, 2, 3, 4, 5, 6]
WINDOW_SIZE = 600
TRAIN_STRIDE = 600
TEST_STRIDE = 600
EPOCHS = 60
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-3
MIXUP_PROB = 0.3
MIXUP_ALPHA = 0.2
LABEL_SMOOTHING = 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_subject(subject_id):
    subject_dir = os.path.join(BASE_PATH, f"DB2_s{subject_id}")
    paths = []
    for root, _, files in os.walk(subject_dir):
        paths += [os.path.join(root, f) for f in files if f.endswith(".mat")]
    if not paths:
        raise FileNotFoundError(subject_dir)
    emg, labels, reps = [], [], []
    for p in sorted(paths):
        m = sio.loadmat(p)
        emg.append(compress_semg(m["emg"]))
        labels.append(m["restimulus"])
        reps.append(m["rerepetition"])
    return np.vstack(emg), np.vstack(labels), np.vstack(reps)


def make_windows(subject_id, stride):
    emg, labels, reps = load_subject(subject_id)
    return create_segmented_windows(
        emg, labels, reps, ALL_REPS,
        window_size=WINDOW_SIZE, stride=stride,
        valid_label_min=1, valid_label_max=49,
    )


def standardize(train, test):
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (train - mean) / std, (test - mean) / std


def tensorize(x):
    return torch.tensor(x, dtype=torch.float32).permute(0, 2, 1)


def mixup(xt, xf, y):
    lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
    idx = torch.randperm(y.size(0), device=y.device)
    return lam * xt + (1-lam) * xt[idx], lam * xf + (1-lam) * xf[idx], y, y[idx], lam


def evaluate(model, loader):
    model.eval(); pred_all=[]; y_all=[]; seg_all=[]
    with torch.no_grad():
        for xt, xf, y, seg in loader:
            logits = model(xt.to(DEVICE), xf.to(DEVICE))
            pred_all.append(logits.argmax(1).cpu().numpy())
            y_all.append(y.numpy()); seg_all.append(seg.numpy())
    pred = np.concatenate(pred_all); y = np.concatenate(y_all); seg = np.concatenate(seg_all)
    smooth = segmented_causal_majority_vote(pred, seg, window_size=5)
    return {
        "Raw_Acc": accuracy_score(y, pred),
        "Raw_F1": f1_score(y, pred, average="macro", zero_division=0),
        "Causal5_Acc": accuracy_score(y, smooth),
        "Causal5_F1": f1_score(y, smooth, average="macro", zero_division=0),
    }


def run_fold(held_out):
    seed = 3000 + held_out; seed_all(seed)
    train_t=[]; train_f=[]; train_y=[]
    for sid in SUBJECTS:
        if sid == held_out: continue
        xt, xf, y, _ = make_windows(sid, TRAIN_STRIDE)
        train_t.append(xt); train_f.append(xf); train_y.append(y-1)
    xtr_t=np.concatenate(train_t); xtr_f=np.concatenate(train_f); ytr=np.concatenate(train_y)
    xte_t, xte_f, yte, seg = make_windows(held_out, TEST_STRIDE); yte = yte-1
    xtr_t, xte_t = standardize(xtr_t, xte_t)
    xtr_f, xte_f = standardize(xtr_f, xte_f)

    tr = TensorDataset(tensorize(xtr_t), tensorize(xtr_f), torch.tensor(ytr, dtype=torch.long))
    te = TensorDataset(tensorize(xte_t), tensorize(xte_f), torch.tensor(yte, dtype=torch.long), torch.tensor(seg, dtype=torch.long))
    train_loader=DataLoader(tr,batch_size=BATCH_SIZE,shuffle=True,num_workers=2,pin_memory=torch.cuda.is_available())
    test_loader=DataLoader(te,batch_size=BATCH_SIZE,shuffle=False,num_workers=2,pin_memory=torch.cuda.is_available())

    model=EdgeNetTF(num_classes=49).to(DEVICE)
    opt=optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    sched=optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=20,T_mult=2)
    criterion=nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    for _ in range(EPOCHS):
        model.train()
        for xt, xf, y in train_loader:
            xt=xt.to(DEVICE); xf=xf.to(DEVICE); y=y.to(DEVICE); opt.zero_grad()
            if random.random() < MIXUP_PROB:
                mt,mf,ya,yb,lam=mixup(xt,xf,y); logits=model(mt,mf)
                loss=lam*criterion(logits,ya)+(1-lam)*criterion(logits,yb)
            else:
                loss=criterion(model(xt,xf),y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        sched.step()
    result=evaluate(model,test_loader); result["Held_Out_Subject"]=held_out
    del model, tr, te, train_loader, test_loader, xtr_t, xtr_f, xte_t, xte_f
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--start",type=int,default=1); parser.add_argument("--end",type=int,default=40); parser.add_argument("--output",default="results/revision/loso_subject_level.csv"); args=parser.parse_args()
    rows=[]
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    for sid in range(args.start,args.end+1):
        print(f"LOSO fold S{sid}",flush=True); rows.append(run_fold(sid)); pd.DataFrame(rows).to_csv(out,index=False)
    df=pd.DataFrame(rows); print(df); print(df[["Raw_Acc","Raw_F1","Causal5_Acc","Causal5_F1"]].agg(["mean",lambda x:x.std(ddof=1)]))

if __name__ == "__main__":
    main()
