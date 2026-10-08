"""Step-size sweep for baseline FxLMS: finds the largest well-behaved mu (Part 1)."""
import numpy as np
from fxlms import *
P = np.load('P.npy'); g = np.load('g.npy'); ghat = np.load('ghat.npy')
rng = np.random.default_rng(1); K = 40; ks = rng.integers(0, 2048, K); Pk = P[:, ks]
nb = 600; x = rng.standard_normal(PRE + nb*100)
print('mu     status  atten_dB   conv(rho=.4)  conv(rho=.1)   [blocks, mean+-std over %d positions]' % K)
for mu in (0.5, 1, 2, 4, 8, 16, 24, 32, 40, 48):
    with np.errstate(all='ignore'):
        W, E, D = fxlms(Pk, g, ghat, x, mu, nblocks=nb); m = block_mse(E)
    if not np.isfinite(m).all() or m[-40:].mean() > D.mean():
        print('%5.1f  DIVERGED' % mu); continue
    att = attenuation_db(D[-40:].mean(0), m[-40:].mean(0))
    c4 = np.array([convergence_time(m[:, k], 0.4) for k in range(K)])
    c1 = np.array([convergence_time(m[:, k], 0.1) for k in range(K)])
    print('%5.1f  stable  %5.1f      %5.1f+-%.1f    %5.1f+-%.1f' % (mu, att.mean(), np.nanmean(c4), np.nanstd(c4), np.nanmean(c1), np.nanstd(c1)))
