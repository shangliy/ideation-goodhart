# Direction bias (stage 12): (a) share of higher-direction hypotheses, direct vs measure-first; (b) measure-first minus direct under the
# original metric, the corrected metric (SV3) and the threshold-free score R. Data: experiments/harness/s12_results.json, s12_paper.json, run caches.
import os,sys,json,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); H=os.path.join(HERE,'..','experiments','harness'); sys.path.insert(0,H); os.chdir(H)
import s12
SURF,INK,INK2,GRID='#fcfcfb','#0b0b0b','#52514e','#e4e3df'; BLUE,ORANGE,AQUA='#2a78d6','#eb6834','#1baf7a'
CELLS=[('nb','sonnet','NB201\nSonnet'),('nb','opus','NB201\nOpus'),('fc','sonnet','HPO\nSonnet'),('fc','opus','HPO\nOpus')]
M3=json.load(open('s12_results.json'))['M3']; PP=json.load(open('s12_paper.json'))
def hi_share(sp,gen,arm):
    C=[c for r in s12.arm_runs(sp,gen,arm) for c in s12.claims(r,False)]; return sum(c[4]>0 for c in C)/len(C)
fig,(a,b)=plt.subplots(1,2,figsize=(7.0,2.5),facecolor=SURF,gridspec_kw=dict(width_ratios=[1,1.55]))
for ax in (a,b):
    ax.set_facecolor(SURF); ax.grid(axis='y',color=GRID,linewidth=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors=INK2,labelsize=7.5,length=0)
x=np.arange(len(CELLS))
for i,(sp,g,_) in enumerate(CELLS):
    d,m=hi_share(sp,g,'BASE'),hi_share(sp,g,'A')
    a.plot([i,i],[d,m],color=INK2,linewidth=2,zorder=2)
    a.scatter([i],[d],s=36,facecolor=SURF,edgecolor=INK,linewidth=1.5,zorder=3); a.scatter([i],[m],s=36,color=INK,edgecolor=SURF,linewidth=1.5,zorder=3)
    a.annotate(f'{m:.2f}',(i,m),xytext=(7,0),textcoords='offset points',fontsize=7,color=INK2,va='center')
a.set_xticks(x); a.set_xticklabels([c[2] for c in CELLS]); a.set_ylim(0,0.7); a.set_xlim(-0.5,3.7)
a.set_title('(a) Share of "higher" hypotheses',fontsize=8.5,color=INK,loc='left')
a.scatter([],[],s=30,facecolor=SURF,edgecolor=INK,label='direct'); a.scatter([],[],s=30,color=INK,label='measure-first')
a.legend(frameon=False,fontsize=7,labelcolor=INK2,loc='lower left',ncol=2,handletextpad=0.2,columnspacing=0.8)
S=[('original $\\Delta$SV',ORANGE,[M3[f'{sp}_{g}']['sens']['old@50'] for sp,g,_ in CELLS]),
   ('corrected $\\Delta$SV$_3$',BLUE,[M3[f'{sp}_{g}']['d'] for sp,g,_ in CELLS]),
   ('$\\Delta R$ (post hoc)',AQUA,[PP[f'{sp}_{g}']['A']['dR'] for sp,g,_ in CELLS])]
w=0.26
for k,(lab,col,v) in enumerate(S):
    xs=x+(k-1)*w
    b.bar(xs,v,width=w-0.03,color=col,label=lab,zorder=3)
    for xx,vv in zip(xs,v): b.annotate(f'{vv:+.2f}',(xx,vv),xytext=(0,2 if vv>=0 else -2),textcoords='offset points',ha='center',va='bottom' if vv>=0 else 'top',fontsize=6,color=INK2)
b.axhline(0,color=INK2,linewidth=0.8,zorder=2)
b.set_xticks(x); b.set_xticklabels([c[2] for c in CELLS]); b.set_ylim(-2.35,2.2)
b.set_title('(b) Measure-first minus direct, per run',fontsize=8.5,color=INK,loc='left')
b.legend(frameon=False,fontsize=7,labelcolor=INK2,loc='upper right',ncol=1,handletextpad=0.4,borderaxespad=0.2)
fig.tight_layout(w_pad=1.5)
for out in (os.path.join(HERE,'direction.pdf'),os.path.join(HERE,'direction.png')): fig.savefig(out,dpi=200,facecolor=SURF)
print('wrote figures/direction.pdf')
