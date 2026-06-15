import os, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from daltonlens import simulate
from daltonlens.simulate import Deficiency
SIM=simulate.Simulator_Machado2009()
DEF={"d":Deficiency.DEUTAN,"p":Deficiency.PROTAN}
HW=(240,320)  # H,W
def L(p): return np.asarray(Image.open(p).convert("L").resize((HW[1],HW[0]),Image.BILINEAR),dtype=np.float32)/255.
def RGB(p): return np.asarray(Image.open(p).convert("RGB").resize((HW[1],HW[0]),Image.BILINEAR),dtype=np.uint8)
def find_orig(t,stem):
    for sp in ("val","train"):
        for e in (".jpeg",".jpg",".png",".bmp"):
            p=f"data/etocvd_{t}_sigma3p0/splits_fold5/fold_1/{sp}/images/{stem}{e}"
            if os.path.exists(p): return p
    return None
def cvdsim(rgb,t): return SIM.simulate_cvd(rgb,DEF[t],1.0).astype(np.uint8)

picks={}
for t in ("d","p"):
    gdir=f"data/etocvd_{t}_recolor_opt/guided_lam0.15"
    cand=[]
    for gp in glob.glob(gdir+"/*.png"):
        stem=os.path.splitext(os.path.basename(gp))[0]
        dpath=f"data/etocvd_{t}_recolor_opt/daltonize/{stem}.png"
        npath=f"data/etocvd_normal_sm_sigma3p0/{stem}.png"
        spath=f"data/etocvd_{t}_scvd_perfold_newgt/{stem}.png"
        op=find_orig(t,stem)
        if op and os.path.exists(dpath) and os.path.exists(npath) and os.path.exists(spath):
            M=np.clip(L(npath)-L(spath),0,None); cand.append((float(M.sum()),stem,op,dpath,gp,npath,spath))
    cand.sort(reverse=True)
    picks[t]=cand[:2]
    print(t,"picked:",[c[1] for c in picks[t]])

rows=[("d",c) for c in picks["d"]]+[("p",c) for c in picks["p"]]
cols=["Original","CVD simulation","Full daltonization","Proposed","Proposed (CVD sim)","Target map $M$"]
fig,axes=plt.subplots(len(rows),6,figsize=(8.6,1.5*len(rows)+0.3))
for r,(t,(mass,stem,op,dpath,gp,npath,spath)) in enumerate(rows):
    orig=RGB(op); prop=RGB(gp); dalt=RGB(dpath)
    M=np.clip(L(npath)-L(spath),0,None); M=M/(M.max()+1e-8)
    panels=[orig, cvdsim(orig,t), dalt, prop, cvdsim(prop,t), M]
    for c,img in enumerate(panels):
        ax=axes[r,c]
        if c==5: ax.imshow(img,cmap="inferno")
        else: ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        if r==0: ax.set_title(cols[c],fontsize=7)
    axes[r,0].set_ylabel(f"{'Deutan' if t=='d' else 'Protan'}\n{stem[:14]}",fontsize=6,rotation=90,va="center")
fig.tight_layout(pad=0.3,h_pad=0.4,w_pad=0.15)
os.makedirs("/tmp/crabuild/images",exist_ok=True)
fig.savefig("/tmp/crabuild/images/fig_qualitative.png",dpi=300,bbox_inches="tight")
print("fig_qualitative.png written")
