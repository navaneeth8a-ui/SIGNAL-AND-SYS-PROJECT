"""Trial harness: random start position, jump to another random position at block 100 (paper Sec. 4.1)."""
import numpy as np
from fxlms import *
P=np.load('P.npy'); g=np.load('g.npy'); ghat=np.load('ghat.npy')
NB, SW = 400, 100

def run_trial(mu, seed, update=None):
    """Returns (on_mse, off_mse): per-block MSE with ANC on / off, arrays of length NB.
    `update` is reserved for Part 3 (a latent-update function); default = plain block NLMS-FxLMS."""
    rng = np.random.default_rng(seed)
    k1, k2 = rng.choice(P.shape[1], 2, replace=False)
    x = rng.standard_normal(PRE + NB*100)
    W, E, D = fxlms(P[:, [k1]], g, ghat, x, mu, nblocks=NB, P2=P[:, [k2]], switch=SW)
    return block_mse(E)[:, 0], D[:, 0], (k1, k2)

def summarize(on, off, rho=0.4):
    """on/off: (trials, NB). Returns dict of per-trial metrics."""
    ntr = on.shape[0]
    c_init = np.array([convergence_time(on[i], rho) for i in range(ntr)])
    c_sw   = np.array([convergence_time(on[i], rho, start=SW) for i in range(ntr)]) 
    att = attenuation_db(off[:, -40:].mean(1), on[:, -40:].mean(1))
    return dict(conv_initial=c_init, conv_after_switch=c_sw, atten_db=att)
