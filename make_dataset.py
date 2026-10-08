import numpy as np, time
from fxlms import *
P=np.load('P.npy'); g=np.load('g.npy'); ghat=np.load('ghat.npy'); Ww=np.load('W_wiener.npy')
L=512
G=np.zeros((2*L-1,L))
for k in range(L): G[k:k+L,k]=g
def att_db(W):                       # exact steady-state attenuation for white noise
    pp=np.vstack([P,np.zeros((L-1,P.shape[1]))]); r=pp+G@W
    return 10*np.log10((pp**2).sum(0)/(r**2).sum(0))
MU=16.0; NB=3000
rng=np.random.default_rng(42); x=rng.standard_normal(PRE+NB*100)
t0=time.time(); W=np.zeros((L,2048)); 
W,_,_=fxlms(P,g,ghat,x,MU,nblocks=NB,store_e=False)
print('FxLMS on 2048 positions, mu=%g, %d blocks: %.0fs'%(MU,NB,time.time()-t0))
a=att_db(W); aw=att_db(Ww)
rel=np.linalg.norm(W-Ww,axis=0)/np.linalg.norm(Ww,axis=0)
print('attenuation of converged FxLMS filters: min %.1f  median %.1f  max %.1f dB'%(a.min(),np.median(a),a.max()))
print('attenuation of Wiener optimum         : min %.1f  median %.1f  max %.1f dB'%(aw.min(),np.median(aw),aw.max()))
print('relative distance ||W_fx - W_wiener||/||W_wiener||: median %.3f max %.3f'%(np.median(rel),rel.max()))
np.save('W.npy',W.T.astype(np.float32))      # dataset: (2048, 512)
np.save('att_fxlms_converged.npy',a)
