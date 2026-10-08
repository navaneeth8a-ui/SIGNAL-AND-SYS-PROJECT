"""Check the vectorised block FxLMS against a plain, sample-by-sample reference implementation."""
import numpy as np
from fxlms import *
P=np.load('P.npy'); g=np.load('g.npy'); L=512; B=100; nb=6; mu=8.0; eps=1e-6
rng=np.random.default_rng(0); x=rng.standard_normal(PRE+nb*B); p=P[:,700]
W,E,D=fxlms(p[:,None],g,g,x,mu,B=B,nblocks=nb)
# reference
xhat=np.convolve(x,g)[:len(x)]; w=np.zeros(L); y=np.zeros(nb*B); e_ref=np.zeros(nb*B)
dfull=np.convolve(x,p)[:len(x)]
for b in range(nb):
    gsum=np.zeros(L)
    for i in range(B):
        n=PRE+b*B+i
        y[b*B+i]=w@x[n-L+1:n+1][::-1]
        ypast=y[:b*B+i+1]; anti=sum(g[j]*ypast[b*B+i-j] for j in range(L) if b*B+i-j>=0)
        e=dfull[n]+anti; e_ref[b*B+i]=e
        xh=xhat[n-L+1:n+1][::-1]; gsum+=e*xh/(eps+xh@xh)
    w=w-mu*gsum/B
print('max |e_fast - e_ref| over %d samples: %.2e (signal rms %.3f)'%(nb*B,np.abs(E.reshape(-1)-e_ref).max(),e_ref.std()))
assert np.allclose(E.reshape(-1),e_ref,atol=1e-10); print('PASS')
