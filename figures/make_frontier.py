# Precision vs surprising-valid (SV) per harness, one panel per generator. Data: experiments/harness/s3_results.json.
import json,os,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE=os.path.dirname(os.path.abspath(__file__))
R=json.load(open(os.path.join(HERE,'..','experiments','harness','s3_results.json')))['harnesses']
SURF,INK,INK2,GRID='#fcfcfb','#0b0b0b','#52514e','#e4e3df'; BLUE,ORANGE='#2a78d6','#eb6834'
BASES={'s2 BASE','s1(0, 0, 0, 0)','BASE'}
LAB={'s2 BASE':'direct','s2 final_J':'judge-optimized','s2 final_V':'validity-optimized','s2 final_SV2':'SV-optimized',
     'BASE':'direct','final_J':'judge-optimized','final_V':'validity-optimized','SV_prior_avoid':'prior-avoid select','SV_cross_domain':'cross-domain search'}
OFF={'s2 final_SV2':(0,9,'center'),'s2 final_V':(6,-14,'left'),'SV_prior_avoid':(-8,-4,'right'),'SV_cross_domain':(8,-4,'left')}
fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharey=True,facecolor=SURF)
for ax,(g,title) in zip(axes,[('sonnet','Generator: Sonnet (13 harnesses)'),('opus','Generator: Opus (5 harnesses)')]):
    ax.set_facecolor(SURF)
    for k,v in R[g].items():
        b=k in BASES
        ax.scatter(v['prec'],v['sv'],s=70 if b else 46,color=ORANGE if b else BLUE,edgecolor=SURF,linewidth=2,zorder=3,marker='D' if b else 'o')
        if k in LAB:
            dx,dy,ha=OFF.get(k,(6,4,'left')); ax.annotate(LAB[k],(v['prec'],v['sv']),xytext=(dx,dy),textcoords='offset points',fontsize=9,color=INK2,ha=ha)
    ax.set_title(title,fontsize=11,color=INK,loc='left'); ax.set_xlabel('precision (share of claims that hold)',color=INK2,fontsize=10)
    ax.grid(color=GRID,linewidth=0.8); ax.set_axisbelow(True); ax.set_xlim(0,1.05); ax.set_ylim(0,2)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors=INK2,labelsize=9,length=0)
axes[0].set_ylabel('SV per run (true and low-prior claims)',color=INK2,fontsize=10)
fig.legend(handles=[Line2D([],[],marker='D',ls='',color=ORANGE,label='direct generation'),Line2D([],[],marker='o',ls='',color=BLUE,label='other harness')],
           loc='upper right',frameon=False,fontsize=9,labelcolor=INK2,ncol=2)
fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig(os.path.join(HERE,'frontier.png'),dpi=160,facecolor=SURF)
