#!/usr/bin/env python3
"""Generate a representative time-domain -> FFT illustration for the revision.

Uses the same sign-preserving logarithmic compression and direct FFT as the main
experiments. Edit BASE_PATH before running.
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from src.edgenettf.preprocessing import compress_semg

BASE_PATH='/kaggle/input/datasets/quddusikashaf/ninapro-db2'; SUBJECT=1; FS=2000; CHANNEL=0; WINDOW=600


def main():
 d=os.path.join(BASE_PATH,f'DB2_s{SUBJECT}'); paths=[]
 for root,_,files in os.walk(d): paths += [os.path.join(root,f) for f in files if f.endswith('.mat')]
 emg=[]; labels=[]; reps=[]
 for p in sorted(paths):
  m=sio.loadmat(p); emg.append(m['emg']); labels.append(m['restimulus']); reps.append(m['rerepetition'])
 emg=compress_semg(np.vstack(emg)); labels=np.vstack(labels).reshape(-1); reps=np.vstack(reps).reshape(-1)
 selected=None
 for rep in [2,5]:
  idx=np.where(rep==reps)[0]
  for block in np.split(idx,np.where(np.diff(idx)>1)[0]+1):
   if len(block)<WINDOW: continue
   bl=labels[block]; cps=np.where(np.diff(bl)!=0)[0]+1
   for seg in np.split(block,cps):
    if len(seg)>=WINDOW and 1<=int(labels[seg[0]])<=49:
     selected=seg[:WINDOW]; break
   if selected is not None: break
  if selected is not None: break
 sig=emg[selected,CHANNEL]; t=np.arange(WINDOW)/FS*1000
 spec=np.fft.fft(sig); logmag=np.log1p(np.abs(spec[:WINDOW//2])); freq=np.fft.fftfreq(WINDOW,d=1/FS)[:WINDOW//2]
 fig,ax=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
 ax[0].plot(t,sig); ax[0].set_xlabel('Time (ms)'); ax[0].set_ylabel('Compressed amplitude'); ax[0].set_title('(a) Time-domain sEMG')
 ax[1].plot(freq,logmag); ax[1].set_xlabel('Frequency (Hz)'); ax[1].set_ylabel('Log magnitude'); ax[1].set_title('(b) FFT log-magnitude')
 fig.savefig('EdgeNetTF_Time_to_FFT.png',dpi=300,bbox_inches='tight')

if __name__=='__main__': main()
