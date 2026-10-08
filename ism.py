"""Image Source Model (Allen & Berkley 1979) for a shoebox room, NumPy only.
Mirrors pyroomacoustics: Sabine-derived uniform absorption, 1/(4*pi*d) spreading,
windowed-sinc fractional delays (81 taps)."""
import numpy as np

C = 343.0
FS = 16000
ROOM = np.array([6.0, 6.2, 3.0])
RT60 = 0.15
L = 512

def sabine_absorption(room=ROOM, rt60=RT60, c=C):
    V = np.prod(room); S = 2*(room[0]*room[1] + room[0]*room[2] + room[1]*room[2])
    return 24*np.log(10.0)*V/(c*S*rt60)

def rir(src, mic, room=ROOM, absorption=None, fs=FS, c=C, n=L, fdl=81):
    if absorption is None: absorption = sabine_absorption(room)
    r = np.sqrt(1.0 - absorption)                       # wall reflection coefficient
    src = np.asarray(src, float); mic = np.asarray(mic, float)
    jmax = int(np.ceil((n + fdl) / fs * c / (2*room.min()))) + 1
    j = np.arange(-jmax, jmax + 1)
    axes = []
    for a in range(3):
        co, rf = [], []
        for q in (0, 1):
            co.append((1 - 2*q)*src[a] + 2*j*room[a]); rf.append(np.abs(j - q) + np.abs(j))
        axes.append((np.concatenate(co), np.concatenate(rf)))
    X, Y, Z = np.meshgrid(axes[0][0], axes[1][0], axes[2][0], indexing='ij')
    RX, RY, RZ = np.meshgrid(axes[0][1], axes[1][1], axes[2][1], indexing='ij')
    d = np.sqrt((X-mic[0])**2 + (Y-mic[1])**2 + (Z-mic[2])**2).ravel()
    nref = (RX + RY + RZ).ravel()
    t = d / c * fs                                       # delay in samples (fractional)
    keep = t < n + fdl // 2
    d, nref, t = d[keep], nref[keep], t[keep]
    amp = r**nref / (4*np.pi*d)
    h = np.zeros(n)
    k = np.arange(-(fdl//2), fdl//2 + 1)
    idx = np.floor(t)[:, None].astype(int) + k[None, :]
    off = idx - t[:, None]
    taps = np.sinc(off) * (0.5*(1 + np.cos(2*np.pi*off/fdl))) * (np.abs(off) <= fdl/2)
    vals = amp[:, None]*taps
    ok = (idx >= 0) & (idx < n)
    np.add.at(h, idx[ok], vals[ok])
    return h

SEC_SRC = np.array([3.0, 2.5, 1.5])
ERR_MIC = np.array([4.5, 3.0, 1.5])
LINE_A = np.array([1.5, 1.0, 1.0]); LINE_B = np.array([3.0, 2.0, 2.0])

def primary_positions(num=2048):
    t = np.linspace(0, 1, num)[:, None]
    return LINE_A[None, :] + t*(LINE_B - LINE_A)[None, :]

if __name__ == '__main__':
    print('absorption', sabine_absorption())
    g = rir(SEC_SRC, ERR_MIC); print('g first arrival (samples):', np.argmax(np.abs(g)), 'expected', np.linalg.norm(SEC_SRC-ERR_MIC)/C*FS)
    pos = primary_positions()
    print('spacing (mm):', 1000*np.linalg.norm(pos[1]-pos[0]), 'min dist to mic:', np.linalg.norm(pos-ERR_MIC, axis=1).min(), 'speaker dist:', np.linalg.norm(SEC_SRC-ERR_MIC))
