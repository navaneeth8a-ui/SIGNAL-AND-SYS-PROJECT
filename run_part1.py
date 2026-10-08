import numpy as np, json, time
from trials import *
MUS=(8,16,24,32); NTR=50
res={}; curves={}
for mu in MUS:
    t0=time.time(); on=[];off=[]
    for s in range(NTR):
        a,b,_=run_trial(mu,s); on.append(a); off.append(b)
    on=np.array(on); off=np.array(off); r=summarize(on,off)
    curves[mu]=(on.mean(0),off.mean(0))
    # also convergence measured on the 50-trial averaged curve (what the paper plots)
    avg=on.mean(0)
    res[mu]={k:(float(np.nanmean(v)),float(np.nanstd(v))) for k,v in r.items()}
    res[mu]['conv_initial_avgcurve']=float(convergence_time(avg,0.4))
    res[mu]['conv_switch_avgcurve']=float(convergence_time(avg,0.4,start=SW))
    print('mu=%2d  initial conv %.0f+-%.0f | after switch %.0f+-%.0f | avg-curve %.0f / %.0f | atten %.1f+-%.1f dB  (%.0fs)'%(mu,*res[mu]['conv_initial'],*res[mu]['conv_after_switch'],res[mu]['conv_initial_avgcurve'],res[mu]['conv_switch_avgcurve'],*res[mu]['atten_db'],time.time()-t0))
np.save('curves.npy',{mu:np.array(c) for mu,c in curves.items()},allow_pickle=True)
json.dump({str(k):v for k,v in res.items()},open('results.json','w'),indent=1)
