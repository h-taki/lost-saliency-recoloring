import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows=list(csv.DictReader(open("paper/tables/raw/recolor_pareto_coords.csv")))
def pts(t,m):
    r=[x for x in rows if x["type"]==t and x["method"]==m]
    r=sorted(r,key=lambda x:(float(x["lam"]) if x["lam"] else 0))
    return [float(x["naturalness_cost"]) for x in r],[float(x["distinguish_gain"]) for x in r],[x["lam"] for x in r]
styles={"guided":("o","-","#1f77b4","Learned $M$ (proposed)"),
        "normalguide":("s","--","#ff7f0e","Normal-saliency guide"),
        "uniform":("^",":","#2ca02c","Uniform guide")}
fig,axes=plt.subplots(2,1,figsize=(3.3,4.4),sharex=False)
for ax,t,lab in [(axes[0],"d","(a) Deutan"),(axes[1],"p","(b) Protan")]:
    for m,(mk,ls,c,leg) in styles.items():
        x,y,lams=pts(t,m)
        ax.plot(x,y,ls,marker=mk,color=c,ms=4,lw=1.2,label=leg)
        # annotate lambda=0.15 on guided
        if m=="guided":
            for xi,yi,l in zip(x,y,lams):
                if l=="0.15":
                    ax.annotate(r"$\lambda{=}0.15$",(xi,yi),textcoords="offset points",xytext=(6,-2),fontsize=6)
    dx,dy,_=pts(t,"daltonize")
    ax.plot(dx,dy,marker="*",color="#d62728",ms=11,ls="none",label="Full daltonization")
    ax.set_title(lab,fontsize=8,loc="left")
    ax.set_ylabel("Distinguishability gain",fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(True,alpha=0.3,lw=0.4)
    ax.annotate("better",(0.02,0.93),xycoords="axes fraction",fontsize=6,style="italic",color="gray")
axes[1].set_xlabel("Naturalness cost (CIEDE2000; lower is better)",fontsize=7)
axes[0].legend(fontsize=5.6,loc="upper right",framealpha=0.9)
fig.tight_layout(pad=0.4)
import os; os.makedirs("/tmp/crabuild/images",exist_ok=True)
fig.savefig("/tmp/crabuild/images/fig_pareto.pdf",bbox_inches="tight")
print("fig_pareto.pdf written")
