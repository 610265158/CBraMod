"""Real TUEV waveform, phase folding, and a traceable 16-sample inset.

Uses the training example and preprocessing in make_lossless_temporal_folding.
Outputs vector PDF and a PNG preview; no synthetic EEG is used.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from make_lossless_temporal_folding import load_tuev_example, TUEV_CHANNELS, TUEV_LABELS


def main():
    eeg, folded, label = load_tuev_example(4)
    # Deterministic illustrative training window: highest energy, 16 samples,
    # aligned to a four-sample group. This is not a model attribution.
    energy = (eeg.reshape(16, 250, 4)**2).sum(-1)
    scores = np.stack([energy[:, i:i+4].sum(1) for i in range(247)], axis=1)
    ch, group = np.unravel_index(np.argmax(scores), scores.shape)
    start = int(group * 4)
    values = eeg[ch, start:start+16]
    local = values.reshape(4, 4).T
    assert np.array_equal(local, folded[ch*4:ch*4+4, group:group+4])
    assert np.array_equal(folded.reshape(16,4,250).transpose(0,2,1).reshape(16,1000), eeg)
    phase = ['#2166AC','#D97706','#138A72','#9B4C96']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':10,
                         'axes.titlepad':19,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7})
    fig = plt.figure(figsize=(10.0,6.4),facecolor='white')
    gs = fig.add_gridspec(2,2,left=.13,right=.94,bottom=.15,top=.84,
                          hspace=.85,wspace=.35,height_ratios=[1.3,1])
    a,b,c,d = [fig.add_subplot(gs[i,j]) for i,j in [(0,0),(0,1),(1,0),(1,1)]]
    scale = max(float(np.percentile(np.abs(eeg),99)),1e-6)*2.8
    time=np.arange(1000)/200
    for k in range(16):
        a.plot(time,k-eeg[k]/scale,color=phase[0] if k==ch else '#536578',lw=.55)
    a.axvspan(start/200,(start+16)/200,color='#E7A026',alpha=.25)
    a.set(xlim=(0,5),ylim=(15.9,-.9),xlabel='Time (s)',yticks=range(16),yticklabels=TUEV_CHANNELS)
    a.set_title('(a) Real EEG: 16 × 1,000',loc='left',fontweight='bold')
    a.tick_params(axis='y',length=0,labelsize=6.5)
    a.text(.02,1.02,'Shared amplitude scale; traces vertically offset',transform=a.transAxes,fontsize=7,color='#596675')
    vmax=float(np.max(np.abs(eeg)))
    norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
    im=b.imshow(folded,aspect='auto',cmap='RdBu_r',norm=norm,interpolation='nearest')
    for k in range(1,16): b.axhline(k*4-.5,color='white',lw=.4)
    b.set(yticks=np.arange(16)*4+1.5,yticklabels=TUEV_CHANNELS,xlabel='Column w (four samples per column)',xticks=[0,50,100,150,200,249])
    b.tick_params(axis='y',length=0,labelsize=6.5)
    b.set_title('(b) Same data folded: 64 × 250',loc='left',fontweight='bold')
    b.add_patch(Rectangle((group-.5,ch*4-.5),4,4,fill=False,ec='#E7A026',lw=1.8))
    b.text(.02,1.02,'Each channel occupies four consecutive phase rows',transform=b.transAxes,fontsize=7,color='#596675')
    cb=fig.colorbar(im,ax=b,fraction=.026,pad=.02);cb.ax.tick_params(labelsize=6);cb.set_label('Scaled EEG amplitude',fontsize=7)
    x=np.arange(16)
    c.plot(x,values,color='#52606E',lw=1)
    for p,col in enumerate(phase):
        ix=x[p::4];c.scatter(ix,values[p::4],c=col,s=25,zorder=4)
    span=max(float(np.ptp(values)),1e-4)
    for i,v in enumerate(values):c.annotate(str(i),(i,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7,color=phase[i%4])
    for edge in [3.5,7.5,11.5]:c.axvline(edge,color='#B9C4CF',ls='--',lw=.8)
    c.set(xlim=(-.7,15.7),ylim=(min(values)-span*.15,max(values)+span*.3),xticks=[0,4,8,12,15],xlabel=f'Local index i (original t = {start} + i)',ylabel='Scaled EEG amplitude')
    c.set_title(f'(c) Zoom: {TUEV_CHANNELS[ch]}, 16 real samples',loc='left',fontweight='bold')
    c.text(.0,1.02,'Numbers identify samples; colors identify phase i mod 4',transform=c.transAxes,fontsize=7,color='#596675')
    d.imshow(local,cmap='RdBu_r',norm=norm,aspect='auto',interpolation='nearest')
    for p in range(4):
        for w in range(4):
            i=4*w+p
            d.text(w,p,f'{i}',ha='center',va='center',fontsize=11,fontweight='bold',color=phase[p],
                   bbox=dict(boxstyle='round,pad=.22',facecolor='white',edgecolor='none',alpha=.95))
    d.set(xticks=range(4),xticklabels=[f'{group+w}' for w in range(4)],yticks=range(4),yticklabels=[f'p = {p}' for p in range(4)],xlabel='Folded column w')
    for p,tick in enumerate(d.get_yticklabels()):tick.set_color(phase[p])
    d.set_xticks(np.arange(-.5,4),minor=True);d.set_yticks(np.arange(-.5,4),minor=True);d.grid(which='minor',color='white',lw=1.2);d.tick_params(which='minor',length=0)
    d.set_title('(d) Folded samples and a 3 × 3 kernel',loc='left',fontweight='bold')
    d.add_patch(Rectangle((-.5,-.5),3,3,fill=False,ec='#E69F00',lw=2.5,zorder=8,clip_on=False))
    d.plot(1,1,marker='s',markersize=25,markerfacecolor='none',
           markeredgecolor='#E69F00',markeredgewidth=1.6,zorder=9)
    d.text(.5,-.34,r'$3\times3$ kernel: center 5; samples 0–2, 4–6, 8–10',
           transform=d.transAxes,ha='center',fontsize=8,color='#815B00')
    assert np.array_equal(np.arange(16).reshape(4,4).T[:3,:3],
                          [[0,4,8],[1,5,9],[2,6,10]])
    d.text(0,1.02,'Numbers identify the same samples shown in (c)',transform=d.transAxes,fontsize=7,color='#596675')
    fig.add_artist(ConnectionPatch(xyA=(15.8,np.mean(values)),coordsA=c.transData,xyB=(-.65,1.5),coordsB=d.transData,arrowstyle='->',mutation_scale=12,color='#667085',lw=1.2))
    fig.text(.51,.29,'P = 4',ha='center',fontsize=9,fontweight='bold')
    fig.suptitle('Lossless phase-interleaved folding on real EEG',fontsize=12,fontweight='bold',y=.97)
    fig.text(.13,.025,r'Each column groups four consecutive samples; each row follows one phase: $I[cP+p,w]=X[c,wP+p]$.',fontsize=9)
    for ax in [a,c]:ax.spines[['top','right']].set_visible(False)
    target=Path(__file__).resolve().parent/'folding_overview'
    fig.savefig(target.with_suffix('.pdf'))
    fig.savefig(target.with_suffix('.png'),dpi=220)
    plt.close(fig)
    print(f'Validated exact fold/unfold; channel={ch}, window=[{start},{start+16}); saved {target}')

if __name__=='__main__':main()
