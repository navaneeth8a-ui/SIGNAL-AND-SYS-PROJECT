import numpy as np, json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ism import *
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
NAVY,TEAL,AMBER,CORAL='#0B2A3C','#1C9AA8','#F2A33A','#D9534F'
P=np.load('P.npy'); g=np.load('g.npy'); W=np.load('W.npy'); pos=np.load('positions.npy')
curves=np.load('curves.npy',allow_pickle=True).item(); res=json.load(open('results.json'))
t_ms=np.arange(512)/FS*1000

# ---- Fig 1: setup
fig=plt.figure(figsize=(12,4.2))
ax=fig.add_subplot(1,3,1,projection='3d')
ax.plot(pos[:,0],pos[:,1],pos[:,2],color=TEAL,lw=3,label='Primary source region')
ax.scatter(*ERR_MIC,c='k',marker='o',s=50,label='Error mic'); ax.scatter(*SEC_SRC,c=AMBER,marker='*',s=140,label='Secondary speaker')
ax.set_xlim(0,6);ax.set_ylim(0,6.2);ax.set_zlim(0,3);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_zlabel('z (m)',labelpad=2);ax.legend(fontsize=7,loc='upper left');ax.set_title('Room 6×6.2×3 m, RT60 0.15 s')
ax=fig.add_subplot(1,3,2)
ax.plot(t_ms,g,color=AMBER,lw=1.2,label='g (secondary)'); 
for k,c in ((0,TEAL),(2047,CORAL)): ax.plot(t_ms,P[:,k],color=c,lw=1,alpha=.9,label='p, position %d'%k)
ax.set_xlabel('time (ms)');ax.set_ylabel('amplitude');ax.set_title('Impulse responses (512 taps, 16 kHz)');ax.legend(fontsize=8)
ax=fig.add_subplot(1,3,3); f=np.fft.rfftfreq(1024,1/FS)/1000
ax.plot(f,20*np.log10(np.abs(np.fft.rfft(g,1024))),color=AMBER,label='|G|'); ax.plot(f,20*np.log10(np.abs(np.fft.rfft(P[:,1000],1024))),color=TEAL,label='|P| (pos 1000)')
ax.set_xlabel('frequency (kHz)');ax.set_ylabel('dB');ax.set_title('Frequency responses');ax.legend(fontsize=8)
fig.tight_layout(w_pad=3);fig.savefig('figures/fig1_setup.png',dpi=140);plt.close()

# ---- Fig 2: dataset / manifold
Wc=W-W.mean(0); U,S,Vt=np.linalg.svd(Wc,full_matrices=False); var=S**2/np.sum(S**2); cum=np.cumsum(var)
n99=int(np.searchsorted(cum,0.99))+1; n999=int(np.searchsorted(cum,0.999))+1; n9999=int(np.searchsorted(cum,0.9999))+1
print('PCA: comps for 99%% var = %d, 99.9%% = %d, 99.99%% = %d; var in 32 comps = %.5f; 1 comp=%.3f 3 comps=%.4f'%(n99,n999,n9999,cum[31],cum[0],cum[2]))
Z=U[:,:3]*S[:3]
fig=plt.figure(figsize=(12,4.2))
ax=fig.add_subplot(1,3,1)
for i,(k,c) in enumerate(zip((0,512,1024,1536,2047),plt.cm.viridis(np.linspace(0,.9,5)))): ax.plot(t_ms,W[k]/np.abs(W).max()*0.9+ (4-i)*1.0,color=c,lw=.8)
ax.set_yticks([]);ax.set_xlabel('tap time (ms)');ax.set_title('Converged filters w\n(top→bottom: source position 0…2047)',fontsize=10)
ax=fig.add_subplot(1,3,2)
loc=[]
for c in range(64,2048-64,128):
    X=W[c-32:c+32]-W[c-32:c+32].mean(0); sv=np.linalg.svd(X,compute_uv=False); loc.append((sv**2/np.sum(sv**2))[:40])
loc=np.mean(loc,0); ax.semilogy(np.arange(1,41),var[:40],'o-',color=CORAL,ms=3,label='global PCA (all 2048 filters)'); ax.semilogy(np.arange(1,41),np.maximum(loc,1e-9),'o-',color=TEAL,ms=3,label='local PCA (64 neighbours)')
ax.set_ylim(1e-7,1);ax.set_xlabel('principal component');ax.set_ylabel('explained variance ratio');ax.set_title('Globally curved, locally low-dimensional');ax.legend(fontsize=8)
ax=fig.add_subplot(1,3,3,projection='3d'); sc=ax.scatter(Z[:,0],Z[:,1],Z[:,2],c=np.arange(2048),cmap='viridis',s=3); ax.set_title('Filters on top-3 PCs: one curve\n(1 parameter = source position)'); ax.set_xlabel('PC1');ax.set_ylabel('PC2');ax.set_zlabel('PC3',labelpad=-2)
fig.tight_layout(w_pad=2.5);fig.savefig('figures/fig2_dataset_manifold.png',dpi=140);plt.close()
json.dump({'pcs_99':n99,'pcs_999':n999,'pcs_9999':n9999,'var_first_comp':float(var[0]),'var_first3':float(cum[2]),'var_first32':float(cum[31])},open('pca.json','w'))

# ---- Fig 3: error curves (like paper Fig. 8/9)
fig,ax=plt.subplots(figsize=(8,4.2)); tt=(np.arange(400)+0.5)*0.00625
for mu,c in zip((8,16,24,32),(CORAL,AMBER,TEAL,NAVY)):
    ax.plot(tt,curves[mu][0]*1e3,color=c,lw=1.6,label='FxLMS, μ=%d'%mu)
ax.axvline(100*0.00625,color='gray',ls=':'); ax.text(100*0.00625+0.02,ax.get_ylim()[1]*0.9,'source jumps',color='gray',fontsize=8)
ax.set_xlabel('time (s)');ax.set_ylabel('mean squared error (×1e-3)');ax.set_title('Baseline FxLMS: 50-trial averaged error signal');ax.legend(fontsize=8)
fig.tight_layout();fig.savefig('figures/fig3_error_curves.png',dpi=140);plt.close()

# ---- Fig 4: tradeoff
mus=[8,16,24,32]; fig,axs=plt.subplots(1,2,figsize=(10,3.8))
ci=[res[str(m)]['conv_initial'] for m in mus]; cs=[res[str(m)]['conv_after_switch'] for m in mus]; at=[res[str(m)]['atten_db'] for m in mus]
x=np.arange(4); axs[0].bar(x-.2,[c[0] for c in ci],.4,yerr=[c[1] for c in ci],color=TEAL,label='from start',capsize=3); axs[0].bar(x+.2,[c[0] for c in cs],.4,yerr=[c[1] for c in cs],color=AMBER,label='after jump',capsize=3)
axs[0].set_xticks(x);axs[0].set_xticklabels(['μ=%d'%m for m in mus]);axs[0].set_ylabel('convergence time (blocks, ρ=0.4)');axs[0].legend(fontsize=8);axs[0].set_title('Larger step → faster')
axs[1].bar(x,[a[0] for a in at],yerr=[a[1] for a in at],color=NAVY,capsize=3); axs[1].axhline(15.8,color=CORAL,ls='--'); axs[1].text(-0.4,16.2,'Wiener optimum ≈15.8 dB',color=CORAL,fontsize=8)
axs[1].set_xticks(x);axs[1].set_xticklabels(['μ=%d'%m for m in mus]);axs[1].set_ylabel('steady-state attenuation (dB)');axs[1].set_ylim(0,19);axs[1].set_title('…but misadjustment grows near the stability limit')
fig.tight_layout();fig.savefig('figures/fig4_stepsize_tradeoff.png',dpi=140);plt.close()
