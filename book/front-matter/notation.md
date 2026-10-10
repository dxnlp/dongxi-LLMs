# Notation and Experimental Conventions

All logarithms are natural unless a base is explicitly stated. Negative
log-likelihood and KL are therefore measured in nats. Token IDs are categorical
indices; their numerical distances do not express linguistic similarity.

| Symbol | Meaning | Typical shape |
|---|---|---|
| $B,n,V,D$ | Batch, sequence length, vocabulary, residual width | Scalar dimensions |
| $X$ | Residual states at all input positions | $[B,n,D]$ |
| $E$ | Input embedding table | $[V,D]$ |
| $W_{\mathrm{out}}$ | Stored vocabulary head; $z_t=h_tW_{\mathrm{out}}^\top$ | $[V,D]$ |
| $h_t,z_t,p_t$ | Contextual state, logits, probabilities | $[D]$, $[V]$, $[V]$ |
| $q_t$ | Target distribution, often one-hot | $[V]$ |
| $H,d_h$ | Attention heads and query/key head width | Scalar dimensions |
| $Q,K,V_{\mathrm{val}}$ | Queries, keys and attention values | $[B,H,n,d_h]$ when value width matches |
| $m_t$ | Valid-target or completion-loss mask | $[B,n]$ |
| $\pi_\theta,\pi_{\mathrm{ref}},\pi_{\mathrm{old}}$ | Current, reference and rollout policies | Conditional distributions |
| $x,y$ | Context and response | Token sequences |
| $r_\phi(x,y),R(x,y)$ | Learned reward and environment/verifier reward | Scalars |
| $\hat A_i,\hat A_{i,t}$ | Response-level and token-level advantage | Response/token arrays |
| $\tau,\Delta t_{\mathrm{update}}$ | Temperature and update duration | Scalar; duration has time units |
| $\beta$ | KL coefficient; its direction is named with the objective | Scalar |
| $\alpha_{\mathrm{LoRA}}$ | LoRA scaling numerator | Scalar |
| $\lambda_{\mathrm{SFT}},\lambda_H,\lambda_{\mathrm{KD}}$ | Chosen-NLL, entropy and distillation weights | Scalars |
| $\delta_t,\epsilon_{\mathrm{adv}}$ | TD residual and advantage-normalization stabilizer | Scalar/token arrays; scalar |
| $G$ | Number of completions per prompt in grouped optimization | Scalar |

Architecture and optimization add the following local quantities:

| Symbol | Meaning | Shape or units |
|---|---|---|
| $L,F$ | Number of decoder blocks; intermediate MLP width | Scalar dimensions |
| $P_{\mathrm{pos}}$ | Baseline learned position table; RoPE has a different mechanism | $[n_{\mathrm{max}},D]$ |
| $A$ | Row-normalized attention weights | $[B,H,n,n]$ |
| $M$ | Additive attention mask: zero for allowed positions, $-\infty$ for forbidden ones | Broadcastable to $[B,H,n,n]$ |
| $\theta,\eta_t$ | Trainable parameters; learning rate at optimizer update $t$ | Tensor collection; scalar |
| $g_t$ | Current parameter gradient | Same shape as its parameter |
| $\beta_1,\beta_2$ | Adam first/second-moment decay coefficients | Scalars in $[0,1)$ |
| $m_t^{\mathrm{Adam}},v_t^{\mathrm{Adam}}$ | Adam gradient and squared-gradient moments | Same shape as their parameter |
| $\lambda_{\mathrm{wd}}$ | Decoupled weight-decay coefficient | Scalar |
| $\epsilon_{\mathrm{norm}},\epsilon_{\mathrm{Adam}},\epsilon_{\mathrm{clip}}$ | Normalization stabilizer, optimizer stabilizer and PPO clipping width | Scalars with different roles |
| $N$ | Count of items or valid targets; the local definition names which | Scalar count |

Preference and policy arguments use:

| Symbol | Meaning | Shape or units |
|---|---|---|
| $\sigma(u)$ | Logistic sigmoid, $1/(1+e^{-u})$ | Scalar probability |
| $\Delta$ | Difference between two reward scores | Scalar |
| $\ell_\theta(x,y)$ | Sum of response-token log probabilities | Scalar in nats |
| $m_\theta$ | Scaled reference-relative DPO pair margin | Scalar |
| $Z(x)$ | Normalizer in the reward-tilted policy; distinct from the logits tensor $Z$ | Positive scalar |
| $s_t,a_t$ | Pre-action prefix/state and sampled next-token action | Token sequence; categorical ID |
| $J(\theta)$ | Expected reward, or the explicitly stated regularized objective | Scalar |
| $\rho$ | Current/behavior likelihood ratio on the declared action or trajectory | Positive scalar |
| $V_\psi(s_t)$ | Critic estimate of expected future return from a pre-action state | Scalar |
| $G_t$ | Return after state $s_t$; distinct from group size $G$ | Scalar reward sum |
| $\gamma,\lambda_{\mathrm{GAE}}$ | Discount factor and GAE residual-weighting factor | Scalars in $[0,1]$ |
| $H(p),D_{\mathrm{KL}}(p\Vert q)$ | Distribution entropy and directional KL divergence | Scalars in nats |
| $k_1,k_2,k_3$ | Sampled KL quantities with estimator-specific contracts | Scalars |

These explicit names help compare chapters. A derivation may write Adam moments
as $m_t,v_t$, the GAE factor as $\lambda$, or weight decay as $\lambda$ after
defining that local meaning. Read the subscript and surrounding definition:
an Adam moment is not a supervision mask, a baseline is not a behavior policy,
and the LayerNorm offset called $\beta$ is not a KL coefficient. Target,
reference and teacher distributions may each be written $q$ locally; their
source, normalization and gradient status are stated at the point of use.

[Chapter 3](../chapters/03-learning-the-next-token.md) develops probability,
entropy and KL from prediction. [Chapter 11](../chapters/11-direct-preference-optimization.md)
uses them for reference-relative preference learning, while
[Chapter 12](../chapters/12-language-generation-as-a-policy.md) develops states,
returns, advantages and behavior-policy ratios. The tables are lookup aids;
the chapters teach why these quantities enter the computation.

The attention value tensor uses a subscript here to avoid confusing it with
vocabulary size. A chapter can use another local symbol after defining it.
Archived figures and Python APIs may call sequence length `T`, head width
`d_k` or `d`, and distillation temperature `T`. Their local mappings are
stated in the chapters; the symbols above do not change code or measurements.
Chapter 2 distinguishes tokenizer entries $V_t$ from model output rows $V_m$.
An output table may be tied to $E$; unique parameter storage is then counted
once even though the parameter has two computational paths.

Python tensors commonly use zero-based positions. At logit position $t$, a
causal next-token model predicts token $t+1$. A training API may accept labels
at their original input positions and shift internally; a manual implementation
typically slices logits and labels explicitly. Every implementation in the book
states which alignment it uses.

The mask $m_t$ selects losses. It does not determine which source positions a
hidden state can attend to. Attention masks control that visibility, while
autograd detachment controls gradient propagation. These three boundaries
must be named separately when describing an intervention.

The same word “step” may refer to a generated token, microbatch, optimizer update
or environment action. Reports use the explicit term. Token budgets name whether
they count valid targets, processed positions, unique source targets or generated
tokens. Throughput names both its numerator and timing boundary.

Measured results are linked to reports. Calculated costs are labeled calculations.
Synthetic fixtures are labeled teaching examples. Proposed Spark protocols remain
proposed until their actual measurements exist. Repeatedly inspected prompts
are development probes; final capability claims need an independent contract.
