"""Block normalized FxLMS (Eq. 4) vectorised over K independent primary positions, plus metrics (Eqs. 15-16)."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy.signal import lfilter

PRE = 1024   # pre-roll samples (2*L): the primary field is already established when ANC starts

def fxlms(P, g, ghat, x, mu, B=100, nblocks=240, W0=None, P2=None, switch=None, eps=1e-6, store_e=True):
    """P: (L,K) primary paths. x: white-noise reference of length >= PRE + nblocks*B (first PRE samples = history).
    If P2 is given, primary paths switch to P2 at block `switch`.
    Returns W (L,K), E (nblocks,B,K) ANC-on error (or None), Dmse (nblocks,K) ANC-off block MSE."""
    L, K = P.shape
    N = nblocks*B
    x = x[:PRE + N]
    xhat = lfilter(ghat, 1.0, x)
    r0 = PRE - (L - 1)
    X  = swv(x,    L)[:, ::-1][r0:r0 + N]      # row n -> [x_n, x_{n-1}, ..., x_{n-L+1}]
    Xh = swv(xhat, L)[:, ::-1][r0:r0 + N]
    pw = np.convolve(xhat**2, np.ones(L))[PRE:PRE + N] + eps      # ||xhat_n||^2
    W = np.zeros((L, K)) if W0 is None else W0.copy()
    Yh = np.zeros((L - 1 + B, K))      # rolling history of the speaker input y
    ii = np.arange(B)[:, None]; mm = np.arange(B + L - 1)[None, :]
    j = ii + L - 1 - mm
    Gm = np.where((j >= 0) & (j < L), g[np.clip(j, 0, L-1)], 0.0)   # (B, B+L-1) Toeplitz of g
    E = np.zeros((nblocks, B, K)) if store_e else None
    Dmse = np.zeros((nblocks, K))
    for b in range(nblocks):
        sl = slice(b*B, (b+1)*B)
        Pc = P2 if (P2 is not None and switch is not None and b >= switch) else P
        y = X[sl] @ W
        Yh[:L-1] = Yh[B:].copy() if B >= L-1 else np.concatenate([Yh[B:L-1], Yh[L-1:]])[-(L-1):]
        Yh[L-1:] = y
        anti = Gm @ Yh
        d = X[sl] @ Pc
        e = d + anti
        Dmse[b] = (d**2).mean(0)
        if store_e: E[b] = e
        W -= mu * (Xh[sl].T @ (e / pw[sl, None])) / B
    return W, E, Dmse

def block_mse(E):
    """E (nblocks,B,K) -> per-block mean squared error (nblocks,K)."""
    return (E**2).mean(axis=1)

def convergence_time(m, rho=0.4, tail=40, start=0):
    """Eq. 15 on a block-MSE sequence m: earliest block k>=start with m_k <= (1+rho)*e_inf^2 (e_inf^2 = mean of last `tail` blocks)."""
    einf = m[-tail:].mean()
    ok = np.where(m[start:] <= (1 + rho)*einf)[0]
    return (ok[0] if len(ok) else np.nan)

def attenuation_db(e_off_mse, e_on_mse):
    """Eq. 16."""
    return 10*np.log10(e_off_mse/e_on_mse)

def wiener(P, g):
    """Closed-form optimum (white-noise reference): argmin_w ||p + g*w||^2, per column of P. Also returns the achievable attenuation (dB)."""
    L = len(g)
    G = np.zeros((2*L-1, L))
    for k in range(L): G[k:k+L, k] = g
    A = G.T @ G; A += 1e-10*np.trace(A)/L*np.eye(L)
    Ppad = np.vstack([P, np.zeros((L-1, P.shape[1]))])
    W = -np.linalg.solve(A, G.T @ Ppad)
    res = Ppad + G @ W
    return W, 10*np.log10((Ppad**2).sum(0)/(res**2).sum(0))
