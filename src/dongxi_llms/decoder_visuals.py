"""Data-backed static teaching figures for Chapter 5 notebooks.

All inputs are detached for plotting, never modified. No random sampling or
model execution happens here. Functions return Matplotlib figures; callers own
display/close. Coordinate projections and illustration-only layouts are labeled.
"""
from functools import wraps

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

BLUE, PURPLE, GREEN, AMBER = '#2563EB', '#7C3AED', '#047857', '#B45309'
INK, MUTED = '#111827', '#64748B'
COLORS = [BLUE, PURPLE, GREEN, AMBER]


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def styled(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        with plt.rc_context({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                             'font.family': 'DejaVu Sans', 'font.size': 11,
                             'axes.titlesize': 12, 'axes.labelsize': 10,
                             'xtick.labelsize': 9, 'ytick.labelsize': 9,
                             'text.color': INK, 'axes.labelcolor': INK,
                             'axes.spines.top': False, 'axes.spines.right': False,
                             'figure.dpi': 110, 'savefig.dpi': 110}):
            return function(*args, **kwargs)
    return wrapper


@styled
def parameter_budget(ledger):
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    ax.barh(list(ledger), list(ledger.values()), color=BLUE)
    ax.invert_yaxis()
    ax.set(xlabel='Unique stored scalar parameters', title='Where the parameter budget goes')
    for i, value in enumerate(ledger.values()):
        ax.text(value, i, f' {value:,}', va='center', fontsize=9)
    ax.margins(x=.18)
    return fig


@styled
def audit_outcomes(reports):
    from matplotlib.colors import ListedColormap
    columns = ['finite', 'causal', 'cache']
    values = np.array([[int(row[key]) for key in columns] for row in reports.values()])
    fig, ax = plt.subplots(figsize=(8, 3.5), layout='constrained')
    ax.imshow(values, cmap=ListedColormap(['#FEE2E2', '#D1FAE5']), vmin=0, vmax=1, aspect='auto')
    ax.set(xticks=range(3), xticklabels=['Finite values', 'Future invariance', 'Cache equivalence'],
           yticks=range(len(reports)), yticklabels=list(reports),
           title='A finite forward pass is not a complete correctness check')
    for i in range(len(reports)):
        for j in range(3):
            ax.text(j, i, 'PASS' if values[i,j] else 'FAIL', ha='center', va='center')
    return fig


def _heat(ax, values, title, xlabel='Feature coordinate', ylabel='Position', limit=None):
    data = array(values)
    if data.ndim != 2:
        raise ValueError('A heatmap needs a two-dimensional matrix')
    limit = max(float(np.abs(data).max()), 1e-12) if limit is None else limit
    image = ax.imshow(data, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto', interpolation='nearest')
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.set_xticks(np.arange(data.shape[1]) if data.shape[1] <= 16 else np.arange(0, data.shape[1], 4))
    ax.set_yticks(np.arange(data.shape[0]) if data.shape[0] <= 16 else np.arange(0, data.shape[0], 4))
    return image


@styled
def embedding_lookup(table, ids):
    table, ids = array(table), array(ids).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(9, 5), width_ratios=[1.2, 1], layout='constrained')
    limit = max(float(np.abs(table).max()), 1e-12)
    im = _heat(axes[0], table, 'One stored row per token ID', ylabel='Token ID', limit=limit)
    _heat(axes[1], table[ids], 'Lookup preserves repeated rows', ylabel='Sequence position', limit=limit)
    axes[1].set_yticks(range(len(ids)), [f'pos {p}: ID {i}' for p, i in enumerate(ids)])
    repeated = next((int(i) for i in ids if np.sum(ids == i) > 1), int(ids[0]))
    for ax, rows in [(axes[0], [repeated]), (axes[1], np.where(ids == repeated)[0])]:
        for row in rows:
            ax.add_patch(Rectangle((-.5, row-.5), table.shape[1], 1, fill=False, edgecolor=GREEN, linewidth=2.5))
    fig.colorbar(im, ax=axes, shrink=.75, label='Embedding coordinate value (shared scale)')
    fig.suptitle(f'ID {repeated} is selected repeatedly, not stored repeatedly')
    return fig


@styled
def matrices(values, titles, title, xlabel='Feature coordinate', ylabel='Position'):
    data = [array(v) for v in values]
    limit = max(max(float(np.abs(v).max()) for v in data), 1e-12)
    fig, axes = plt.subplots(1, len(data), figsize=(min(11, 3.3*len(data)), 3.7), squeeze=False, layout='constrained')
    for ax, value, label in zip(axes[0], data, titles):
        im = _heat(ax, value, label, xlabel, ylabel, limit)
    fig.colorbar(im, ax=list(axes[0]), shrink=.75, label='Signed value (shared scale)')
    fig.suptitle(title)
    return fig


@styled
def lookup_gradients(ids, gradients):
    ids, gradients = array(ids).astype(int).ravel(), array(gradients)
    counts = np.bincount(ids, minlength=gradients.shape[0])
    fig, ax = plt.subplots(figsize=(9, 3.5), layout='constrained')
    positions = np.arange(len(counts))
    ax.bar(positions, gradients[:, 0], color=BLUE, alpha=.75, label='Measured gradient, coordinate 0')
    ax.scatter(positions, counts, marker='x', s=55, color=INK, label='Number of lookup occurrences', zorder=3)
    ax.set(xticks=positions, xlabel='Stored token ID', ylabel='Gradient / occurrence count',
           title='For the sum objective, repeated uses add into one row')
    ax.legend(loc='upper right', fontsize=9)
    return fig


@styled
def heads(q):
    q = array(q)
    if q.ndim != 3:
        raise ValueError('Supply one batch item: [heads,time,features]')
    rows = int(np.ceil(len(q)/2))
    fig, axes = plt.subplots(rows, 2, figsize=(9, 3*rows), squeeze=False, layout='constrained')
    limit = max(float(np.abs(q).max()), 1e-12)
    for h, ax in enumerate(axes.flat):
        if h < len(q):
            im = _heat(ax, q[h], f'Query head {h}', 'Head feature coordinate', 'Token position', limit)
        else:
            ax.set_visible(False)
    fig.colorbar(im, ax=[ax for ax in axes.flat if ax.get_visible()], shrink=.7, label='Query value (shared scale)')
    fig.suptitle('Every head has every token position; feature views differ')
    return fig


@styled
def attention_maps(weights, selected=5):
    weights = array(weights)
    rows = int(np.ceil(len(weights)/2))
    fig, axes = plt.subplots(rows, 2, figsize=(9, 3.3*rows), squeeze=False, layout='constrained')
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad('#E5E7EB')
    t, s = weights.shape[-2:]
    if t != s:
        raise ValueError('This figure expects square, full-sequence causal attention')
    mask = np.triu(np.ones((t, s), dtype=bool), 1)
    for h, ax in enumerate(axes.flat):
        if h >= len(weights):
            ax.set_visible(False); continue
        im = ax.imshow(np.ma.array(weights[h], mask=mask), cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        ax.set(title=f'Head {h}', xlabel='Source / key position', ylabel='Receiver / query position',
               xticks=range(s), yticks=range(t))
        if 0 <= selected < t:
            ax.add_patch(Rectangle((-.5, selected-.5), s, 1, fill=False, edgecolor=AMBER, linewidth=2))
        for i in range(t):
            for j in range(s):
                text = '×' if mask[i,j] else f'{weights[h,i,j]:.2f}'
                ax.text(j, i, text, ha='center', va='center', fontsize=8,
                        color='white' if not mask[i,j] and weights[h,i,j] > .6 else INK)
    fig.colorbar(im, ax=[ax for ax in axes.flat if ax.get_visible()], shrink=.7, label='Attention probability')
    fig.suptitle('Different heads, same causal boundary — × = forbidden future')
    return fig


@styled
def value_mixture(weights, values):
    weights, values = array(weights), array(values)
    contributions = weights[:, None]*values
    mixed = contributions.sum(0)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), width_ratios=[1, 1.4, .8], layout='constrained')
    axes[0].barh(np.arange(len(weights)), weights, color=BLUE)
    axes[0].invert_yaxis()
    axes[0].set(title='Routing weights', xlabel='Probability', ylabel='Source position', xlim=(0,1), yticks=range(len(weights)))
    limit = max(float(np.abs(contributions).max()), float(np.abs(mixed).max()), 1e-12)
    im = _heat(axes[1], contributions, 'Each source contributes a × v', ylabel='Source position', limit=limit)
    _heat(axes[2], mixed[None], 'Sum across sources', ylabel='One output row', limit=limit)
    fig.colorbar(im, ax=list(axes[1:]), shrink=.75, label='Signed feature contribution')
    fig.suptitle('One receiver, one head: probability selects how much content to mix')
    return fig


@styled
def vectors(first, update, title, labels=('Input', 'Update', 'Result')):
    first, update = array(first).ravel()[:2], array(update).ravel()[:2]
    result = first+update
    points = np.vstack(([0,0], first, result))
    fig, ax = plt.subplots(figsize=(6.5, 4.5), layout='constrained')
    for origin, delta, color, label in [(np.zeros(2),first,BLUE,labels[0]),
            (first,update,PURPLE,labels[1]), (np.zeros(2),result,GREEN,labels[2])]:
        ax.add_patch(FancyArrowPatch(origin, origin+delta, arrowstyle='-|>', mutation_scale=16,
                                    color=color, linewidth=2, label=label))
    lo, hi = points.min(0), points.max(0)
    pad = np.maximum(hi-lo, .5)*.35
    ax.set(xlim=(lo[0]-pad[0],hi[0]+pad[0]), ylim=(lo[1]-pad[1],hi[1]+pad[1]),
           xlabel='Coordinate 0', ylabel='Coordinate 1', title=title)
    ax.axhline(0, color='#CBD5E1', linewidth=.7); ax.axvline(0, color='#CBD5E1', linewidth=.7)
    ax.legend(fontsize=9); ax.set_aspect('equal', adjustable='datalim')
    fig.suptitle('First two coordinates only — a projection, not the whole vector')
    return fig


@styled
def features(series, title):
    fig, ax = plt.subplots(figsize=(9, 3.7), layout='constrained')
    for i,(label, values) in enumerate(series.items()):
        data = array(values).ravel()
        ax.plot(np.arange(len(data)), data, marker=['o','s','^','x'][i%4],
                color=COLORS[i%4], label=label, linewidth=1.5, markersize=4)
    ax.axhline(0,color='#CBD5E1',linewidth=.8)
    ax.set(title=title, xlabel='Feature coordinate (one position)', ylabel='Signed feature value')
    ax.legend(fontsize=9)
    return fig


@styled
def activation_curve():
    import torch
    import torch.nn.functional as F
    x = torch.linspace(-4,4,201, dtype=torch.float64)
    fig, axes = plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    for ax, name, y in [(axes[0],'GELU', F.gelu(x)), (axes[1],'SiLU gate', F.silu(x))]:
        ax.plot(array(x),array(x),color=MUTED,linestyle='--',label='Identity')
        ax.plot(array(x),array(y),color=PURPLE,label=name,linewidth=2)
        ax.axhline(0,color='#CBD5E1',linewidth=.7)
        ax.set(title=name, xlabel='Input coordinate value', ylabel='Output coordinate value')
        ax.legend(fontsize=9)
    fig.suptitle('Nonlinear transforms — neither curve is a probability distribution')
    return fig


@styled
def decoder_route(cfg):
    labels = [('Token IDs','[B,T]'),('Token + position','[B,T,D]'),
              ('Pre-norm attention','[B,T,D] + update'),('Pre-norm MLP','[B,T,D] + update'),
              (f'{cfg.layers} blocks total','[B,T,D]'),('Final norm + head',f'[B,T,{cfg.vocab}] logits')]
    fig, ax = plt.subplots(figsize=(9,5),layout='constrained')
    ax.set(xlim=(-.6,2.6),ylim=(-.7,1.7)); ax.axis('off')
    coordinates=[(0,1),(1,1),(2,1),(2,0),(1,0),(0,0)]
    for i, ((name,shape),(x,y)) in enumerate(zip(labels,coordinates)):
        ax.add_patch(FancyBboxPatch((x-.43,y-.23),.86,.46,boxstyle='round,pad=.03',
                    facecolor='#F1F5F9',edgecolor=COLORS[min(i//2,3)],linewidth=1.4))
        ax.text(x,y+.07,name,ha='center',va='center',fontsize=10)
        ax.text(x,y-.10,shape,ha='center',va='center',fontsize=9,color=MUTED)
        if i:
            px,py=coordinates[i-1]
            direction=np.array([x-px,y-py]); offset=direction*np.array([.47,.27])
            ax.add_patch(FancyArrowPatch(np.array([px,py])+offset,np.array([x,y])-offset,
                                        arrowstyle='->',mutation_scale=15,color=INK))
    fig.suptitle('The baseline decoder: token positions persist while feature operations change')
    return fig


@styled
def learning(history, final_loss, before, after, targets):
    before, after, targets = array(before), array(after), array(targets).astype(int)
    def softmax(z):
        z = z-z.max(-1,keepdims=True); p=np.exp(z); return p/p.sum(-1,keepdims=True)
    before, after = softmax(before), softmax(after)
    fig, axes=plt.subplots(1,3,figsize=(11,3.8),width_ratios=[1.2,1,1],layout='constrained')
    losses = np.r_[array(history),final_loss]
    axes[0].semilogy(np.arange(len(losses)),losses,color=BLUE)
    axes[0].set(title='Same batch, fixed budget',xlabel='Completed optimizer updates',ylabel='Cross-entropy (nats, log scale)')
    for ax,p,title in [(axes[1],before,'Before training'),(axes[2],after,'After training')]:
        im=ax.imshow(p,vmin=0,vmax=1,cmap='Blues',aspect='auto',interpolation='nearest')
        ax.scatter(targets,np.arange(len(targets)),marker='x',color=AMBER,label='Observed next-token label')
        ax.set(title=title,xlabel='Candidate token ID',ylabel='Input position',xticks=range(0,p.shape[1],2))
    axes[2].legend(loc='upper center',bbox_to_anchor=(.5,-.25),fontsize=8)
    fig.colorbar(im,ax=list(axes[1:]),shrink=.7,label='Probability')
    fig.suptitle('Learning the declared batch — not evidence of general language capability')
    return fig


@styled
def rotary(q, rotated, positions):
    q,rotated,positions=array(q),array(rotated),array(positions)
    fig,axes=plt.subplots(1,2,figsize=(9,4.3),layout='constrained')
    for ax, vector, rot, p in zip(axes,q[:2],rotated[:2],positions[:2]):
        a,b=vector[:2],rot[:2]; radius=float(np.linalg.norm(a))
        circle=plt.Circle((0,0),radius,fill=False,color='#CBD5E1',linestyle='--');ax.add_patch(circle)
        for value,color,label in [(a,BLUE,'Before rotation'),(b,PURPLE,'After rotation')]:
            ax.add_patch(FancyArrowPatch((0,0),value,arrowstyle='-|>',mutation_scale=16,color=color,label=label,linewidth=2))
        span=max(radius*1.3,.1)
        ax.set(xlim=(-span,span),ylim=(-span,span),aspect='equal',title=f'Position {int(p)}',xlabel='Pair coordinate 0',ylabel='Pair coordinate 1')
        ax.legend(fontsize=8)
    fig.suptitle('One coordinate pair rotates without changing its length')
    return fig


@styled
def grouped_heads(query_heads=4, kv_heads=2):
    if query_heads%kv_heads: raise ValueError('Query heads must divide into KV groups')
    fig,ax=plt.subplots(figsize=(7.5,4.5),layout='constrained')
    ax.set(xlim=(-.6,1.6),ylim=(-.7,query_heads-.3));ax.axis('off')
    groups=query_heads//kv_heads
    for i in range(query_heads):
        j=i//groups; target=j*groups+(groups-1)/2
        ax.text(0,i,f'Query head {i}',ha='center',va='center',bbox=dict(boxstyle='round,pad=.4',facecolor='#EFF6FF',edgecolor='none'))
        ax.add_patch(FancyArrowPatch((.22,i),(.73,target),arrowstyle='->',mutation_scale=12,color=COLORS[j%4]))
    for j in range(kv_heads):
        y=j*groups+(groups-1)/2
        ax.text(1,y,f'KV head {j}',ha='center',va='center',bbox=dict(boxstyle='round,pad=.5',facecolor='#F5F3FF',edgecolor='none'))
    ax.invert_yaxis()
    fig.suptitle('Queries stay distinct; source projections are shared within a group')
    return fig


@styled
def costs(rows):
    fig,axes=plt.subplots(1,3,figsize=(10,3.8),layout='constrained')
    names=[f"{r['kv_heads']} KV" for r in rows]
    for ax,key,title,unit in zip(axes,['parameters','logical_kv_bytes','dense_forward_matmul_flops'],
            ['Stored parameters','Compact cache payload','Dense matmul estimate'],['Parameter count','Bytes','Forward FLOPs']):
        values=[r[key] for r in rows]
        ax.bar(names,values,color=BLUE)
        ax.set(title=title,ylabel=unit,xlabel='KV head count',ylim=(0,max(values)*1.25))
        for i,value in enumerate(values):ax.text(i,value,f'{value:,}',ha='center',va='bottom',fontsize=9)
    fig.suptitle('Three different accounting questions — no measured speedup claimed')
    return fig


@styled
def recurrence(ledger):
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    names=['One pass\nshared block','Two passes\nshared block','Two blocks\nindependent']
    for ax,key,title,color in [(axes[0],'stored_block_parameters','Stored block parameters',BLUE),
                               (axes[1],'block_applications','Block applications per forward',PURPLE)]:
        values=[r[key] for r in ledger]
        ax.bar(names,values,color=color)
        ax.set(title=title,ylabel='Count',ylim=(0,max(values)*1.25))
        for i,value in enumerate(values):ax.text(i,value,str(value),ha='center',va='bottom')
    fig.suptitle('Weight sharing saves storage; extra applications still perform computation')
    return fig
