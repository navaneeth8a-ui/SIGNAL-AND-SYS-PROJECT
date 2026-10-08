import numpy as np, time
from ism import *
from fxlms import wiener
t0=time.time()
pos = primary_positions(2048)
P = np.stack([rir(p, ERR_MIC) for p in pos], axis=1)       # (512, 2048)
g = rir(SEC_SRC, ERR_MIC)
np.save('positions.npy', pos); np.save('P.npy', P); np.save('g.npy', g); np.save('ghat.npy', g.copy())
print('RIRs built in %.1fs'%(time.time()-t0), P.shape)
Ww, att = wiener(P, g)
np.save('W_wiener.npy', Ww)
print('Wiener-optimal attenuation dB: min %.1f median %.1f max %.1f'%(att.min(), np.median(att), att.max()))
print('energy of g in 512 taps tail (last 64):', (g[-64:]**2).sum()/(g**2).sum())
