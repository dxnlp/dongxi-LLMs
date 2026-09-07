"""Static architecture schematics, not activation values or measured timings.

Symbols are deliberately generic: B=batch, T=positions, D=model width,
F=MLP width, V=vocabulary size, Hq/Hkv=head counts, d=head width.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

from .decoder_visuals import styled, BLUE, PURPLE, GREEN, AMBER, INK, MUTED


def canvas(title, width=9, height=6, bounds=(-1,9,-1,9)):
    fig, ax = plt.subplots(figsize=(width,height), layout='constrained')
    ax.set(xlim=bounds[:2], ylim=bounds[2:]); ax.axis('off')
    fig.suptitle(title, fontsize=14)
    return fig, ax


def box(ax, x, y, title, shape='', width=2.25, height=.95, active=True, role=BLUE):
    ax.add_patch(FancyBboxPatch((x-width/2,y-height/2),width,height,
                boxstyle='round,pad=.04',facecolor='#EFF6FF' if active else '#F8FAFC',
                edgecolor=role if active else '#CBD5E1',linewidth=2 if active else 1))
    ax.text(x,y+(.15 if shape else 0),title,ha='center',va='center',fontsize=11,color=INK)
    if shape: ax.text(x,y-.20,shape,ha='center',va='center',fontsize=10,color=MUTED)


def arrow(ax, start, end, via=(), color=INK, dashed=False):
    points=[start,*via,end]
    for i,(a,b) in enumerate(zip(points,points[1:])):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->' if i==len(points)-2 else '-',
            mutation_scale=13,color=color,linewidth=1.6,linestyle='--' if dashed else '-'))


def add(ax,x,y,symbol='+',active=True):
    ax.add_patch(Circle((x,y),.22,facecolor='white',edgecolor=AMBER if active else MUTED,linewidth=2))
    ax.text(x,y,symbol,ha='center',va='center',fontsize=15)


@styled
def model_map(focus='embeddings', modern=False):
    """The same five-stage orientation map, with this lesson's component lit."""
    valid={'embeddings','attention','residual','norm','mlp','assembly','training','modern','positions','gqa','recurrence'}
    if focus not in valid: raise ValueError(f'Unknown architecture focus: {focus}')
    fig,ax=canvas('Where this lesson sits in the decoder',8.5,6.2,(-1,8,-.2,9.9))
    stages=[('Token IDs','[B,T]',9),
            ('Token embedding' if modern else 'Token + position embeddings','[B,T,D]',7.1),
            ('Shared block, R applications' if focus=='recurrence' else 'Stack of decoder blocks','[B,T,D] → [B,T,D]',5.2),
            ('Final RMSNorm' if modern else 'Final LayerNorm','[B,T,D]',3.3),
            ('Vocabulary output head','[B,T,V] logits',1.4)]
    selected={1} if focus=='embeddings' else {0,1,2,3,4} if focus in ('assembly','training') else {2,3} if focus in ('norm','modern') else {2}
    for i,(title,shape,y) in enumerate(stages):
        box(ax,2.6,y,title,shape,width=4.1,active=i in selected,role=AMBER if i in selected else BLUE)
        if i: arrow(ax,(2.6,stages[i-1][2]-.52),(2.6,y+.52))
    notes={
        'embeddings':'Current focus\nTwo lookups join\nbefore the first block.',
        'attention':'Current focus\nRetrieve from the\nallowed prefix.',
        'residual':'Current focus\nSkip paths surround\nboth block sublayers.',
        'norm':'Current focus\nNormalize branch inputs;\nnormalize once at the end.',
        'mlp':'Current focus\nTransform features\nat each position.',
        'assembly':'Current focus\nConnect every layer\nwithout losing the axes.',
        'training':'Current focus\nThe observed next token\nsupervises these logits.',
        'modern':'Current focus\nRMSNorm + SwiGLU\nreplace baseline pieces.',
        'positions':'Current focus\nRoPE rotates Q and K\ninside attention.',
        'gqa':'Current focus\nFewer KV heads;\nqueries stay distinct.',
        'recurrence':'Optional variant\nReuse the same weights\nacross block applications.'}
    ax.text(5.2,5.2,notes[focus],va='center',ha='left',fontsize=11,linespacing=1.7)
    ax.text(2.6,.2,'Architecture schematic • shapes are symbolic, not timing or activation values',ha='center',fontsize=9,color=MUTED)
    return fig


@styled
def embedding_detail():
    fig,ax=canvas('Embedding close-up: two lookups meet through addition',9,6,(-1,9,-1,8.8))
    for x,label,shape in [(1.5,'Token IDs','[B,T]'),(6.5,'Position indices','[T]')]:
        box(ax,x,7.5,label,shape)
    box(ax,1.5,5.5,'Token table E','[V,D] → lookup → [B,T,D]',width=3.7,role=BLUE)
    box(ax,6.5,5.5,'Position table P','[max_length,D] → [T,D]',width=3.7,role=PURPLE)
    arrow(ax,(1.5,7),(1.5,6));arrow(ax,(6.5,7),(6.5,6))
    add(ax,4,3.6)
    arrow(ax,(1.5,5),(3.78,3.6),via=((1.5,3.6),),color=BLUE)
    arrow(ax,(6.5,5),(4.22,3.6),via=((6.5,3.6),),color=PURPLE)
    ax.text(6.4,4.25,'Broadcast across batch',ha='center',fontsize=9,color=PURPLE)
    box(ax,4,1.9,'Initial states X','[B,T,D]',role=GREEN)
    arrow(ax,(4,3.38),(4,2.4))
    box(ax,4,.1,'First decoder block','Context mixing begins here',width=3.5,active=False)
    arrow(ax,(4,1.4),(4,.6))
    ax.text(4,-.7,'Addition keeps D features; it does not concatenate the two vectors.',ha='center',fontsize=10)
    return fig


@styled
def attention_detail(grouped=False):
    fig,ax=canvas('Attention close-up: routing and value content meet at AV',10,8,(-.8,9,-1.3,10.5))
    box(ax,4,9.7,'Incoming states X','[B,T,D]',width=2.8)
    for x,label,shape in [(1,'W_Q','[D,Hq·d]'),(4,'W_K','[D,Hkv·d]'),(7,'W_V','[D,Hkv·d]')]:
        box(ax,x,7.9,label,shape,role=PURPLE)
        arrow(ax,(4,9.2),(x,8.4),via=((x,9.2),))
    for x,title,shape in [(1,'Split Q heads','[B,Hq,T,d]'),(4,'Split K heads','[B,Hkv,T,d]'),(7,'Split V heads','[B,Hkv,T,d]')]:
        box(ax,x,6.2,title,shape)
        arrow(ax,(x,7.4),(x,6.7))
    box(ax,2.5,4.5,'QKᵀ / √d','[B,Hq,T,T]',width=2.8)
    arrow(ax,(1,5.7),(1.5,5),via=((1,5),))
    arrow(ax,(4,5.7),(3.5,5),via=((4,5),))
    box(ax,2.5,2.8,'Causal mask → softmax','A: [B,Hq,T,T]',width=3.5)
    arrow(ax,(2.5,4),(2.5,3.3))
    box(ax,5,1.1,'Weighted values: AV','[B,Hq,T,d]',width=3)
    arrow(ax,(2.5,2.3),(3.45,1.1),via=((2.5,1.1),))
    arrow(ax,(7,5.7),(6.55,1.1),via=((7,1.1),),color=GREEN)
    ax.text(7.15,3.6,'Value content\nbypasses softmax',ha='left',va='center',fontsize=10,color=GREEN)
    box(ax,5,-.65,'Concatenate heads → W_O','[B,T,Hq·d] → [B,T,D]',width=4.4,role=GREEN)
    arrow(ax,(5,.6),(5,-.15))
    rule='Hq/Hkv query heads share each KV head' if grouped else 'Ordinary MHA: Hq = Hkv = H'
    ax.text(-.55,-.3,rule.replace(' query heads','\nquery heads') if grouped else rule.replace(': ',':\n'),fontsize=10,va='center')
    return fig


@styled
def block_detail(focus='residual', modern=False):
    norm='RMSNorm' if modern else 'LayerNorm'
    mlp='SwiGLU MLP' if modern else 'GELU MLP'
    fig,ax=canvas('One pre-norm decoder block: branch transforms + skip paths',8.8,8,(-.7,8,-.7,10))
    box(ax,2.7,9.3,'Incoming state X','[B,T,D]',active=False,width=3)
    for y,title,kind in [(7.8,norm+' 1','norm'),(6.3,'Multi-head attention','attention'),
                         (3.7,norm+' 2','norm'),(2.2,mlp,'mlp')]:
        box(ax,2.7,y,title,'[B,T,D] → [B,T,D]',width=3.3,
            active=focus in ('assembly',kind) or (modern and kind in ('norm','mlp')),
            role=PURPLE if kind=='norm' else BLUE)
    add(ax,2.7,4.9,active=focus in ('assembly','residual'))
    add(ax,2.7,.8,active=focus in ('assembly','residual'))
    for start,end in [(8.8,8.3),(7.3,6.8),(5.8,5.12),(4.68,4.2),(3.2,2.7),(1.7,1.02)]:
        arrow(ax,(2.7,start),(2.7,end))
    color=AMBER if focus in ('assembly','residual') else MUTED
    arrow(ax,(4.25,9.3),(2.92,4.9),via=((6,9.3),(6,4.9)),color=color)
    arrow(ax,(2.7,4.4),(2.92,.8),via=((6,4.4),(6,.8)),color=color)
    ax.text(6.2,7.1,'Skip X\n(no norm here)',fontsize=10,va='center',color=color)
    ax.text(6.2,2.8,'Skip updated state U',fontsize=10,va='center',rotation=90,color=color)
    ax.text(1.1,4.4,'U',fontsize=12,va='center',color=GREEN)
    arrow(ax,(2.7,.58),(2.7,-.05))
    ax.text(2.7,-.35,'Output Y: [B,T,D]',ha='center',fontsize=11,color=GREEN)
    return fig


@styled
def mlp_detail(gated=False):
    fig,ax=canvas('Positionwise MLP: transform features without mixing positions',9,6,(-.8,8.8,-.8,8.8))
    box(ax,4,8,'Incoming state','[B,T,D]',width=3)
    if gated:
        box(ax,1.5,6.2,'Gate projection','[B,T,F]',role=PURPLE)
        box(ax,6.5,6.2,'Content projection','[B,T,F]',role=BLUE)
        arrow(ax,(4,7.5),(1.5,6.7),via=((1.5,7.5),))
        arrow(ax,(4,7.5),(6.5,6.7),via=((6.5,7.5),))
        box(ax,1.5,4.4,'SiLU','Not a probability',role=PURPLE)
        arrow(ax,(1.5,5.7),(1.5,4.9))
        add(ax,4,2.7,symbol='×')
        arrow(ax,(1.5,3.9),(3.78,2.7),via=((1.5,2.7),),color=PURPLE)
        arrow(ax,(6.5,5.7),(4.22,2.7),via=((6.5,2.7),),color=BLUE)
        ax.text(4,3.5,'Elementwise product',ha='center',fontsize=10)
        box(ax,4,1,'Down projection','[B,T,F] → [B,T,D]',width=3.5,role=GREEN)
        arrow(ax,(4,2.48),(4,1.5))
    else:
        for y,title,shape in [(6,'Up projection + bias','[B,T,D] → [B,T,F]'),
                (4,'GELU','[B,T,F] → [B,T,F]'),(2,'Down projection + bias','[B,T,F] → [B,T,D]')]:
            box(ax,4,y,title,shape,width=3.9,role=PURPLE if y==4 else BLUE)
            arrow(ax,(4,y+1.5),(4,y+.5))
    ax.text(4,-.2,'Same weights at every position • only feature width changes',ha='center',fontsize=10)
    return fig


@styled
def rope_detail():
    fig,ax=canvas('RoPE lives on Q and K, not on the value branch',9,6,(-.8,8.8,-.8,8.8))
    for x,title,shape in [(1,'Projected Q','[B,Hq,T,d]'),(4,'Projected K','[B,Hkv,T,d]'),(7,'Projected V','[B,Hkv,T,d]')]:
        box(ax,x,7.5,title,shape)
    for x,title in [(1,'Rotate Q at positions'),(4,'Rotate K at positions')]:
        box(ax,x,5.5,title,'Same tensor shape',width=2.7,role=PURPLE)
        arrow(ax,(x,7),(x,6))
    ax.text(2.5,6.6,'Position indices set both rotations',ha='center',fontsize=10,color=PURPLE)
    box(ax,2.5,3.5,'Scaled scores → mask → softmax','[B,Hq,T,T]',width=4.6)
    arrow(ax,(1,5),(1.5,4),via=((1,4),))
    arrow(ax,(4,5),(3.5,4),via=((4,4),))
    box(ax,4,1.5,'Mix unrotated values','A × V',width=3.3,role=GREEN)
    arrow(ax,(2.5,3),(4,2),via=((2.5,2),))
    arrow(ax,(7,7),(5.7,1.5),via=((7,1.5),),color=GREEN)
    ax.text(7.2,4.6,'V bypasses\nRoPE',fontsize=10,color=GREEN,va='center')
    ax.text(4,.1,'During decoding: rotate only new Q/K at continuing positions.\nOld cached keys keep their original rotation.',ha='center',fontsize=10)
    return fig


@styled
def training_detail():
    fig,ax=canvas('Training connects prediction error back to shared parameters',10,6,(-1,10,-1,8))
    box(ax,1,6.8,'Input token IDs','[B,T]',width=2.5)
    box(ax,5,6.8,'Decoder forward','Embeddings + blocks + head',width=4)
    arrow(ax,(2.3,6.8),(2.95,6.8))
    box(ax,5,4.8,'Vocabulary logits','[B,T,V]',width=3)
    arrow(ax,(5,6.3),(5,5.3))
    box(ax,1,2.8,'Next-token labels','[B,T] — shifted once',width=3)
    box(ax,5,2.8,'Cross-entropy loss','One scalar L',width=3)
    arrow(ax,(5,4.3),(5,3.3));arrow(ax,(2.55,2.8),(3.45,2.8))
    box(ax,5,.7,'Backward → gradients','One gradient per trainable parameter',width=5)
    arrow(ax,(5,2.3),(5,1.2),color=PURPLE)
    box(ax,.8,.7,'Optimizer step','Learning rate sets scale',width=3)
    arrow(ax,(2.45,.7),(2.35,.7),color=PURPLE)
    arrow(ax,(-.75,.7),(2.95,6.8),via=((-0.75,5.5),(2.65,5.5),(2.65,6.8)),color=AMBER,dashed=True)
    ax.text(.7,5,'Updated parameters\nfor the next forward',ha='center',fontsize=10,color=AMBER)
    return fig


@styled
def recurrence_detail():
    fig,ax=canvas('Recurrent depth: one set of weights, two different applications',10,5,(-1,10,-1,6))
    box(ax,4.5,4.8,'Shared block parameters θ','Stored once',width=3.9,role=PURPLE)
    box(ax,2.5,2.5,'Apply block Fθ','h₀ → h₁',width=2.6)
    box(ax,6.5,2.5,'Apply the same Fθ','h₁ → h₂',width=2.8)
    arrow(ax,(3.1,4.3),(2.5,3),via=((2.5,4.3),),color=PURPLE)
    arrow(ax,(5.9,4.3),(6.5,3),via=((6.5,4.3),),color=PURPLE)
    ax.text(-.5,2.5,'h₀',va='center',fontsize=13)
    arrow(ax,(-.1,2.5),(1.15,2.5))
    arrow(ax,(3.85,2.5),(5.05,2.5))
    ax.text(4.5,2.9,'h₁',ha='center',fontsize=11)
    arrow(ax,(7.95,2.5),(9,2.5));ax.text(9.15,2.5,'h₂',va='center',fontsize=13)
    ax.text(4.5,.8,'Backward sums both uses into the same ∇θ.\nSame weights ≠ same states or K/V cache entries.',ha='center',fontsize=12,linespacing=1.6)
    return fig
