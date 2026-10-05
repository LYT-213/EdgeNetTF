#!/usr/bin/env python3
"""Generate gesture-level DB2 sample/window statistics for Supplementary Table S1.

Counts active raw samples and train/test windows per subject and gesture without
materializing the window arrays. Edit BASE_PATH before running.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio

BASE_PATH='/kaggle/input/datasets/quddusikashaf/ninapro-db2'
WINDOW_SIZE=600; TRAIN_REPS={1,3,4,6}; TEST_REPS={2,5}; TRAIN_STRIDE=60; TEST_STRIDE=600


def load_labels_reps(sid):
 d=os.path.join(BASE_PATH,f'DB2_s{sid}'); L=[]; R=[]
 paths=[]
 for root,_,files in os.walk(d): paths += [os.path.join(root,f) for f in files if f.endswith('.mat')]
 for p in sorted(paths):
  m=sio.loadmat(p); L.append(m['restimulus'].reshape(-1)); R.append(m['rerepetition'].reshape(-1))
 return np.concatenate(L).astype(int),np.concatenate(R).astype(int)


def count_windows(labels,reps,gesture,target_reps,stride):
 mask=(labels==gesture)&np.isin(reps,list(target_reps)); idx=np.where(mask)[0]
 if not len(idx): return 0
 # split whenever samples are non-contiguous or repetition changes
 cut=np.where((np.diff(idx)>1)|(reps[idx[1:]]!=reps[idx[:-1]]))[0]+1
 total=0
 for seg in np.split(idx,cut):
  if len(seg)>=WINDOW_SIZE: total += 1+(len(seg)-WINDOW_SIZE)//stride
 return total


def main():
 rows=[]
 for sid in range(1,41):
  labels,reps=load_labels_reps(sid)
  for g in range(1,50):
   rows.append({'Subject':sid,'Gesture':g,'Raw_Samples':int((labels==g).sum()),'Train_Windows':count_windows(labels,reps,g,TRAIN_REPS,TRAIN_STRIDE),'Test_Windows':count_windows(labels,reps,g,TEST_REPS,TEST_STRIDE)})
 df=pd.DataFrame(rows); out=Path('results/revision/gesture_statistics_subject_level.csv'); out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False)
 summary=df.groupby('Gesture')[['Raw_Samples','Train_Windows','Test_Windows']].agg(['mean',lambda x:x.std(ddof=1)])
 summary.to_csv('results/revision/gesture_statistics_summary.csv'); print(summary)

if __name__=='__main__': main()
