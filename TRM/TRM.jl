### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 5efe92c1-ba22-4b3b-a06f-c038b6ab9dde
begin
	using PlutoUI
	TableOfContents()
end

# ╔═╡ 9bf0c59e-5cc0-40c2-8ece-43cdebb7f44a
begin
	using Flux
	using Flux: onehot, onehotbatch, logitcrossentropy
	using LinearAlgebra
	using Statistics
	using Random
	using Printf
end

# ╔═╡ e5ccc157-574f-4f71-8288-05243441d804
using Zygote

# ╔═╡ e36102b1-9ef1-4c7a-a61a-7cd66a757ec1
md"""
# Understanding the TRM Architecture

The **Tiny Recursion Model (TRM)** represents a significant simplification over the Hierarchical Reasoning Model (HRM). While HRM uses two separate networks operating at different frequencies (inspired by biological arguments about brain hierarchies), TRM achieves better generalization with just **one tiny 2-layer network**.

### Key Insight: Why Two Features (y and z)

TRM maintains two state variables during recursion:
- **`y`** (previously called `zH` in HRM): The current predicted solution/answer
- **`z`** (previously called `zL` in HRM): A latent reasoning feature (like a "chain of thought")

The paper explains why exactly two features are optimal:
1. If we don't pass the previous reasoning `z`, the model forgets *how* it arrived at solution `y`
2. If we don't pass the previous solution `y`, the model would have to store it inside `z`, wasting capacity
3. More features (>2) provide no benefit and hurt generalization

### The Recursion Process

Each "full recursion" consists of:
1. **n latent updates**: `z ← net(x, y, z)` repeated n times (refining reasoning)
2. **1 answer update**: `y ← net(y, z)` once (proposing improved answer)

With deep supervision, this is repeated T times (T-1 without gradients, 1 with gradients), allowing the model to iteratively improve its answer across up to 16 supervision steps.

### Why "Less is More"

Surprisingly, using fewer parameters leads to *better* generalization:
- 2-layer networks outperform 4-layer networks
- Single network outperforms separate fL and fH networks
- This is attributed to reduced overfitting on small datasets (~1000 training examples)
"""

# ╔═╡ 1c2f8368-fc86-11f0-9ea7-c5cd0c983c87
md"""
## Implementation

A Julia/Flux implementation of the **Tiny Recursion Model** from  
*"Less is More: Recursive Reasoning with Tiny Networks"* by Alexia Jolicoeur-Martineau (Samsung SAIL Montréal).

TRM recursively improves a predicted answer `y` using a single tiny 2-layer network.  
It maintains an input `x` (question), prediction `y` (answer), and latent `z` (reasoning),  
applying deep supervision with adaptive computational time (ACT).
"""

# ╔═╡ 5a19623c-a055-4d4f-8671-aa1ff4734983
md"""
### Configuration
"""

# ╔═╡ 19ce7080-77fc-4e86-9508-657b5211cfa0
Base.@kwdef struct TRMConfig
	vocab_size::Int
	seq_len::Int
	hidden_size::Int   = 128
	expansion::Float64 = 4.0
	num_heads::Int     = 4
	H_cycles::Int      = 3    # T  (outer deep-recursion loops)
	L_cycles::Int      = 6    # n  (inner latent-reasoning steps per loop)
	L_layers::Int      = 2    # transformer layers per block
	halt_max_steps::Int = 16  # N_sup
	rms_norm_eps::Float64 = 1e-5
	rope_theta::Float64   = 10000.0
	use_mlp_t::Bool       = false
end;

# ╔═╡ a562dfde-11d9-468a-ac38-228471873ca8
md"""
### RMS Norm (Root Mean Square Normalisation)

**RMSNorm** is a simplified layer normalisation technique used in modern transformers (like LLaMA). Unlike LayerNorm which centers the distribution by subtracting the mean, RMSNorm only rescales by the root mean square:

$$\mathrm{RMSNorm}(\mathbf{x}) =
\gamma \odot \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\lVert \mathbf{x} \rVert^2 + \varepsilon}}$$

This is computationally cheaper and empirically works as well as full LayerNorm for transformer architectures. The TRM paper follows the modern practice of using RMSNorm with no learnable parameters (no gain/bias terms).
"""

# ╔═╡ 127fe2ef-9dbb-435e-9b19-3ea049a2d58d
function rms_norm(x::AbstractArray; eps=1e-5f0)
	v = mean(x .^ 2; dims=1)
	x ./ sqrt.(v .+ eps)
end

# ╔═╡ 8f72baaf-65a2-4a2f-99a7-d36dd283fedf
md"""
### SwiGLU Feed-Forward Network

**SwiGLU** (Swish-Gated Linear Unit) is the MLP activation function used in modern transformers like LLaMA and PaLM. It combines the Swish activation with a gating mechanism:

$$\mathrm{SwiGLU}(\mathbf{x}) = (\mathbf{x} W_a) \odot\bigl((\mathbf{x} W_b) \odot \sigma(\mathbf{x} W_b)\bigr)$$

where ⊙ is element-wise multiplication.

The gating mechanism allows the network to selectively pass information, which has been shown to improve performance over standard ReLU or GELU activations. The expansion factor (default 4.0) controls the intermediate dimension relative to the hidden size.

**In TRM**: The SwiGLU MLP serves as one of two main components in each TRMBlock (the other being attention). During recursion, information flows through both attention and SwiGLU to iteratively refine the latent reasoning.
"""

# ╔═╡ 3c781eb1-8ae9-4329-8978-eef444a0c411
begin
	struct SwiGLU{A,B}
    	gate_up::A
		down::B
	end
	Flux.@layer SwiGLU

	function SwiGLU(din::Int; expansion=4.0)
    inter = cld(round(Int, expansion * din * 2 / 3), 64) * 64
		    SwiGLU(Dense(din => 2inter; bias=false), Dense(inter => din; bias=false))
	end

	function (m::SwiGLU)(x)
	    gu = m.gate_up(x)
		half = size(gu, 1) >> 1
		gate = gu[1:half, :, :]
		up   = gu[half+1:end, :, :]
		m.down(NNlib.sigmoid_fast.(gate) .* gate .* up)
	end
end

# ╔═╡ 8d36c9ec-2d7d-44dc-9b04-b350545c6490
md"""
### Rotary Position Embeddings (RoPE)

**RoPE** encodes position information by rotating query and key vectors in the embedding space. Unlike absolute position embeddings, RoPE has the elegant property that the dot product between rotated queries and keys depends only on their *relative* positions.

For each position $m$ and dimension pair $(i, i+1)$, RoPE applies a 2D rotation:

$$R(\theta) =
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \cos\theta
\end{bmatrix}$$

Then for position m:

$$\mathrm{RoPE\_rotate}\big( x, m \big) =
\big( R(m\,\omega_1) \oplus R(m\,\omega_2) \oplus \cdots \oplus R(m\,\omega_{d/2})\big)\, x$$

**In TRM**: RoPE allows the model to understand positional relationships in input sequences (like Sudoku grids or maze paths) without adding extra parameters. The precomputed cos/sin tables make this efficient during forward passes.
"""

# ╔═╡ 904ff3d6-0831-435f-b0b7-ea194687c475
begin
	struct RoPE
    cos_tab::Matrix{Float32}  # (head_dim, seq_len)
		    sin_tab::Matrix{Float32}
	end

	function RoPE(head_dim::Int, max_len::Int; theta=10000.0)
    	inv = Float32.(1 ./ (theta .^ (collect(0:2:head_dim-1) ./ head_dim)))
		t   = Float32.(0:max_len-1)
		f   = t * inv'                      # (len, hd/2)
		emb = hcat(f, f)                    # (len, hd)
		RoPE(permutedims(cos.(emb)), permutedims(sin.(emb)))
	end

	function apply_rope(q, k, r::RoPE)
	    hd  = size(q, 1)
		seq = size(q, 2)
		h   = hd >> 1
		c = @view r.cos_tab[:, 1:seq]
		s = @view r.sin_tab[:, 1:seq]
		rot(x) = vcat(-x[h+1:end,:,:], x[1:h,:,:])
		q2 = q .* c .+ rot(q) .* s
		k2 = k .* c .+ rot(k) .* s
		q2, k2
	end
end;

# ╔═╡ 1df3c443-4801-49b3-bc69-af5800427537
md"""
### Multi-Head Attention (Bidirectional)

Standard **self-attention** computes weighted combinations of values based on query-key similarity:


$$\mathrm{Attention}^{(h)}(X)
=
\mathrm{softmax}\!\left(
\frac{Q^{(h)} (K^{(h)})^\top}{\sqrt{d_h}}
\right) V^{(h)}$$

**Multi-head attention** runs this in parallel with different learned projections, allowing the model to attend to information from different representation subspaces.

**Key design choices in TRM**:
- **Bidirectional**: Unlike causal (autoregressive) attention, TRM uses full bidirectional attention since it processes fixed-size inputs (like 9×9 Sudoku grids)
- **No bias**: Following modern practice, linear projections have no bias terms
- **RoPE integration**: Position information is injected via rotary embeddings on Q and K

**Paper insight**: The paper found that replacing self-attention with MLPs worked better for small fixed-context tasks (like 9×9 Sudoku), but attention was necessary for larger 30×30 grids (Maze, ARC-AGI).
"""

# ╔═╡ a4019dbb-708e-4304-bea6-8804b6738259
begin
	struct MHA{A,B}
	    qkv::A
		o_proj::B
		nh::Int
		hd::Int
	end
	
	Flux.@layer MHA

	function MHA(D::Int, nh::Int)
    	hd = D / nh |> Int
		MHA(Dense(D => 3D; bias=false), Dense(D => D; bias=false), nh, hd)
	end

	function (m::MHA)(x, rope::RoPE)
		D, S, B = size(x)
		hd, nh = m.hd, m.nh
		qkv = m.qkv(x)                           # (3D, S, B)
		q = reshape(permutedims(reshape(qkv[1:D,:,:],     hd,nh,S,B), (1,3,2,4)), hd,S,nh*B)
		k = reshape(permutedims(reshape(qkv[D+1:2D,:,:], hd,nh,S,B), (1,3,2,4)), hd,S,nh*B)
		v = reshape(permutedims(reshape(qkv[2D+1:3D,:,:],hd,nh,S,B), (1,3,2,4)), hd,S,nh*B)
		q, k = apply_rope(q, k, rope)
		sc = Float32(1/sqrt(hd))
		
		# scores = qTk / sqrt(d)
		scores = NNlib.batched_mul(
					permutedims(q, (2,1,3)), k) .* sc     # (S,S,nh*B)
		w = softmax(scores; dims=2)               # over keys
		out = NNlib.batched_mul(v, permutedims(w, (2,1,3)))  # (hd,S,nh*B)
		m.o_proj(reshape(permutedims(reshape(out, hd,S,nh,B), (1,3,2,4)), D,S,B))
	end
end

# ╔═╡ c4735111-c774-4f5a-8342-8128f5ff139d
md"""
### TRM Block (Attention + MLP with Post-Norm Residuals)

A **TRMBlock** is the fundamental building unit of the TRM network. It combines:
1. **Multi-head Attention** for capturing relationships between positions
2. **SwiGLU MLP** for non-linear feature transformation

**Residual connections with post-norm** (unlike pre-norm in many modern transformers):

$$h = \mathrm{RMSNorm}(h + \mathrm{Attention}(h))$$

$$h = \mathrm{RMSNorm}(h + \mathrm{MLP}(h))$$

**Key insight from the paper**: TRM uses only **2 layers** (2 TRMBlocks), which is surprisingly optimal. Using 4 layers (like HRM) actually *hurts* generalisation due to overfitting on small datasets. This is the "less is more" principle - when data is scarce, smaller networks with more recursion steps outperform larger networks.

The effective depth comes from recursion: with T=3 loops and n=6 inner steps, each supervision step processes through 7×2 = 14 effective layer applications.
"""

# ╔═╡ 6fb42c0c-5d8a-41b5-8f00-daaef59e2bdd
begin
	struct TRMBlock{A,B}
		attn::A
		mlp::B
		eps::Float32
	end
	Flux.@layer TRMBlock

	TRMBlock(cfg::TRMConfig) = TRMBlock(
		MHA(cfg.hidden_size, cfg.num_heads),
		SwiGLU(cfg.hidden_size; expansion=cfg.expansion),
		Float32(cfg.rms_norm_eps))

	function (b::TRMBlock)(h, rope)
		h = rms_norm(h .+ b.attn(h, rope); eps=b.eps)
		h = rms_norm(h .+ b.mlp(h);        eps=b.eps)
	end
end

# ╔═╡ db43fda5-db51-49b0-9412-76d618587a09
md"""
### Reasoning Module (Input Injection + TRMBlock Stack)

The **ReasoningModule** implements the core latent recursion loop from the paper. It:

1. **Injects** the embedded input question $x$ into the hidden state $h$
2. **Applies** a stack of TRMBlocks to refine the representation

This corresponds to the paper's latent update step:

$$z \leftarrow 	\mathrm{net}(x, y, z)$$

**Input injection** is crucial: by adding $x$ to $h$ at each recursion step, the model maintains awareness of the original question. Without this, the model would "forget" what it's trying to solve.

The forward pass iterates: $h = h + \mathrm{injection}$, then $h = \mathrm{TRMBlock}(h)$ for each layer. This is repeated $n$ times (L_cycles) to update the latent reasoning $z$.
"""

# ╔═╡ 022dd229-826d-41d4-aef1-a5fd11cfd2cd
begin
	struct ReasoningModule{T}
		layers::T
	end
	Flux.@layer ReasoningModule

	function (rm::ReasoningModule)(h, injection, rope)
		h = h .+ injection
		for layer in rm.layers
			h = layer(h, rope)
		end
		h
	end
end

# ╔═╡ ee93226c-2d62-41e6-ae4b-3129804a36da
md"""
### TRM Inner (Embeddings + Reasoning + Output Heads)

**TRMInner** is the complete TRM network architecture that combines:

1. **Input Embedding**: Maps discrete tokens to continuous vectors
2. **Language Model Head** ($f_O$): Projects hidden states back to vocabulary logits  
3. **Q-Head**: Predicts whether the current answer is correct (for ACT halting)
4. **RoPE**: Rotary position embeddings for positional information
5. **L_level**: A single linear layer that updates $y$ from $z$ (the answer refinement step)
6. **ReasoningModule**: The stack of TRMBlocks for latent recursion

**The key architectural insight**: TRM uses a **single network** for both:
- Updating latent $z$: $z \leftarrow \mathrm{net}(x, y, z)$ (includes $x$ in input)
- Updating answer $y$: $y \leftarrow \mathrm{net}(y, z)$ (no $x$ in input)

The presence or absence of $x$ in the input naturally tells the network which task to perform. HRM used two separate networks, but TRM achieves better results with one!
"""

# ╔═╡ 941a0f67-da14-434b-a34f-ab951f33a706
begin
	struct TRMInner{E,H,Q,R,L}
		cfg::TRMConfig
		embed::E
		lm_head::H
		q_head::Q
		rope::R
		L_level::L
		y_init::Vector{Float32}
		z_init::Vector{Float32}
	end
	
	Flux.@layer TRMInner trainable=(embed, lm_head, q_head, L_level)

	function TRMInner(cfg::TRMConfig)
		D = cfg.hidden_size
		TRMInner(
			cfg,
			Embedding(cfg.vocab_size => D),
			Dense(D => cfg.vocab_size; bias=false),
			Dense(D => 2),
			RoPE(D / cfg.num_heads |> Int, cfg.seq_len; theta=cfg.rope_theta),
			ReasoningModule(Tuple(TRMBlock(cfg) for _ in 1:cfg.L_layers)),
			randn(Float32, D),
			randn(Float32, D))
	end

	# One full recursion process: n L-steps on z, then 1 step on y.
	function forward_inner(m::TRMInner, x_ids, y, z)
		x_emb = m.embed(x_ids) .* Float32(sqrt(m.cfg.hidden_size))
		for _ in 1:m.cfg.L_cycles
			z = m.L_level(z, y .+ x_emb, m.rope)
		end
		y = m.L_level(y, z, m.rope)
		logits   = m.lm_head(y)             # (V, S, B)
		q_logits = m.q_head(y[:, 1, :])     # (2, B)
		y, z, logits, q_logits
	end

end;

# ╔═╡ 138bf3b3-4d0e-4f1b-bcbb-c1479ad800a0
md"""
### TRM Outer Wrapper (Deep Supervision & ACT)

This is the outer **TRM** struct that wraps the TRMInner network and implements:

#### Deep Supervision
The model doesn't just predict once - it **iteratively improves** its answer over up to $N_{sup}=16$ steps:
1. Start with initial embeddings for $y$ and $z$
2. Run the recursion process to get a prediction
3. Compute loss and update parameters
4. **Carry forward** the updated $y$ and $z$ to the next step
5. Repeat, allowing the model to progressively refine its answer

#### Adaptive Computational Time (ACT)
Training on every sample for 16 steps is wasteful. **ACT** learns when to stop early:
- The Q-head predicts whether the current answer is correct
- If predicted correct, move to the next training example
- This dramatically reduces training time (often <2 steps on average)

#### The Recursion Process (H\_cycles × L\_cycles)
- **H_cycles = T**: Number of outer recursion loops
- **L_cycles = n**: Number of latent updates per loop
- Each loop: n updates to z, then 1 update to y
- First T-1 loops run without gradients; final loop tracks gradients
"""

# ╔═╡ 40a0a544-256b-44ad-8272-d5cca9f751c4
begin
	struct TRM{I}
		    inner::I
	end
	Flux.@layer TRM

	TRM(cfg::TRMConfig) = TRM(TRMInner(cfg))

	"""Deep recursion: T-1 no-grad loops then 1 with grad."""
	function deep_recursion(m::TRM, x_ids, y, z)
		T = m.inner.cfg.H_cycles
		for _ in 1:(T-1)
			y, z = Zygote.ignore_derivatives() do
				yt, zt, _, _ = forward_inner(m.inner, x_ids, y, z)
				yt, zt
			end
		end
		y, z, logits, q_logits = forward_inner(m.inner, x_ids, y, z)
		Zygote.dropgrad(y), Zygote.dropgrad(z), logits, q_logits
	end

	function init_yz(m::TRM, S::Int, B::Int)
		D = m.inner.cfg.hidden_size
		y = repeat(reshape(m.inner.y_init, D, 1, 1), 1, S, B)
		z = repeat(reshape(m.inner.z_init, D, 1, 1), 1, S, B)
		y, z
	end
end  

# ╔═╡ f6bfa719-022a-4aaa-b786-d9f020fa6e7d
const IGNORE_LABEL = -100;

# ╔═╡ 28d10242-2d2c-45b0-88b3-ff58432d55bc


# ╔═╡ 7b2c4eed-dc18-442e-a5d9-0355eef143a7
md"""
## Demo: Identity-copy task

We train a tiny TRM to copy input tokens to output tokens.
This verifies the full pipeline: model construction, forward pass
with deep recursion, loss computation, and gradient-based training.
"""

# ╔═╡ dcd6c8f2-6b64-456d-a24e-5f7b19370a09
function apply_dim1(f, A::Array{Float32,3})
    a, b, c = size(A)
    R = [Float32(f(@view A[:, j, k])) for j in 1:b, k in 1:c]
    return R
end

# ╔═╡ f7bf8738-899e-4f98-b18f-6d54576b64e4
"""
	trm_loss(model, x_ids, labels, y, z)

Perform one deep-supervision step.  Returns `(y, z, loss, metrics)`.
"""
function trm_loss(model::TRM, x_ids, labels, y, z)
	y, z, logits, q_logits = deep_recursion(model, x_ids, y, z)
	V = size(logits, 1)

	# Cross-entropy loss  (ignoring IGNORE_LABEL positions)
	mask = labels .!= IGNORE_LABEL
	safe_labels = ifelse.(mask, labels, ones(Int, size(labels)))
	# One-hot targets: (V, S, B)
	oh = Flux.onehotbatch(vec(safe_labels), 1:V)
	oh3 = reshape(oh, V, size(labels)...)
	lp = Flux.logsoftmax(logits; dims=1)
	per_pos = -sum(oh3 .* lp; dims=1)[1, :, :]   # (S,B)
	per_pos_masked = per_pos .* mask
	n_valid = max.(sum(mask; dims=1), 1)           # (1,B)
	lm_loss = sum(per_pos_masked ./ n_valid)

	# Q-halt loss  (binary CE: target = sequence fully correct)
 	preds = apply_dim1(argmax, logits)
	correct = mask .& (preds .== labels)
	seq_correct = Float32.(sum(correct; dims=1) .== sum(mask; dims=1))  # (1,B)
	q_halt = q_logits[1:1, :]                             # (1,B)
	q_halt_loss = Flux.logitbinarycrossentropy(q_halt, seq_correct)

	loss = lm_loss + 0.5f0 * q_halt_loss

	acc = sum(correct) / max(sum(mask), 1)
	exact_acc = mean(seq_correct)

	y, z, loss, (acc, exact_acc, lm_loss=lm_loss, q_loss=q_halt_loss)
end

# ╔═╡ 703d3823-54a7-4218-9977-8888a891396c
"""
    train_step!(model, opt_state, x_ids, labels; N_sup=16)

One full deep-supervision training step (up to N_sup supervision steps).
Returns final metrics.
"""
function train_step!(model, opt_state, x_ids, labels; N_sup=16)
    S, B = size(x_ids)
    y, z = init_yz(model, S, B)
    total_loss = 0f0
    
    for step in 1:N_sup
        # Compute gradients - only return scalar loss from gradient block
        loss_val, grads = Flux.withgradient(model) do m
            _, _, l, _ = trm_loss(m, x_ids, labels, y, z)
            l
        end
        
        # Update model parameters
        Flux.update!(opt_state, model, grads[1])
        
        # Re-run forward to get updated y, z (no grad)
        y, z, _, q_logits = deep_recursion(model, x_ids, y, z)
        total_loss += Float32(loss_val)
        
        # Check halt (training): halt if q_halt > 0
        halt_logit = q_logits[1:1, :]
        if all(halt_logit .> 0)
            break
        end
    end
    total_loss
end

# ╔═╡ cb7dd025-19c1-4d87-a94f-2d19601ceebd
let
	# ── Tiny config for the copy task ──
	V = 3          # vocab size (tokens 1..12)
	S = 5          # sequence length
	B = 1           # batch size
	D = 8          # hidden size (small for demo)

	cfg = TRMConfig(
		vocab_size   = V,
		seq_len      = S,
		hidden_size  = D,
		expansion    = 4.0,
		num_heads    = 2,
		H_cycles     = 2,   # T  (fewer for speed)
		L_cycles     = 2,   # n
		L_layers     = 2,
		halt_max_steps = 4,  # N_sup (fewer for demo)
		rms_norm_eps = 1e-5,
		rope_theta   = 10000.0,
		use_mlp_t    = false,
	)

	model = TRM(cfg)

	nparams = length(Flux.destructure(model)[1])
	@info "Model parameters" nparams

	# ── Generate task data ──
	Random.seed!(42)
	make_batch() = begin
		x = rand(1:V, S, B)
	#copy sequence:
		(x_ids=x, labels=copy(x))
	#reverse sequence:
	#	(x_ids=x, labels=reverse(x; dims=1)) 
	end

	# ── Train for a few epochs ──
	opt_state = Flux.setup(Adam(1f-3), model)
	losses = Float32[]

	for epoch in 1:20
		batch = make_batch()
		l = train_step!(model, opt_state, batch.x_ids, batch.labels; 
						N_sup=cfg.halt_max_steps)
		push!(losses, l)
		if epoch % 5 == 0
			@info "Epoch $epoch" loss=l
		end
	end

	# ── Evaluate ──
	test = make_batch()
	logits = Float32[] # bring logits into this scope
	S_t, B_t = size(test.x_ids)
	y, z = init_yz(model, S_t, B_t)
	for _ in 1:cfg.halt_max_steps
		y, z, logits, _ = deep_recursion(model, test.x_ids, y, z)
	end
	preds = apply_dim1(argmax, logits)
	
	mask = test.labels .!= IGNORE_LABEL
	acc = sum(mask .& (preds .== test.labels)) / sum(mask)
	@info "Final evaluation" accuracy=acc

	(losses, accuracy=acc)
end

# ╔═╡ 2992e437-b934-40de-9f73-baa7bca4cf18
# ╠═╡ disabled = true
#=╠═╡
#KOR 7Feb2026 from Claude Code to inspect a mini model
let                                                                  
      V, S, B, D = 3, 5, 2, 8                                                                                                      
      cfg = TRMConfig(                                                 
          vocab_size=V, seq_len=S, hidden_size=D, expansion=4.0,
          num_heads=1, H_cycles=2, L_cycles=2, L_layers=2,
          halt_max_steps=4, rms_norm_eps=1e-5, rope_theta=10000.0,
          use_mlp_t=false,
      )
      m = TRM(cfg)

      Random.seed!(99)
      x_ids = rand(1:V, S, B)
      y, z = init_yz(m, S, B)

      # The input to MHA on the first reasoning step
      x_emb = m.inner.embed(x_ids) .* Float32(sqrt(D))
      h = z .+ (y .+ x_emb)   # first injection: h = z + (y + x_emb)

      # Inspect MHA in first TRMBlock
      attn = m.inner.L_level.layers[1].attn
      rope = m.inner.rope
      hd, nh = attn.hd, attn.nh

      qkv = attn.qkv(h)
      q = reshape(permutedims(reshape(qkv[1:D,:,:],     hd,nh,S,B), (1,3,2,4)), hd,S,nh*B)
      k = reshape(permutedims(reshape(qkv[D+1:2D,:,:], hd,nh,S,B), (1,3,2,4)), hd,S,nh*B)
      v = reshape(permutedims(reshape(qkv[2D+1:3D,:,:],hd,nh,S,B), (1,3,2,4)), hd,S,nh*B)
      q, k = apply_rope(q, k, rope)

      sc = Float32(1/sqrt(hd))
      scores = NNlib.batched_mul(permutedims(q, (2,1,3)), k) .* sc
      w = softmax(scores; dims=2)
      out = NNlib.batched_mul(v, permutedims(w, (2,1,3)))
      final = attn.o_proj(reshape(permutedims(reshape(out, hd,S,nh,B), (1,3,2,4)), D,S,B))

      @info "MHA input (h)" size=size(h) h
      @info "Q" size=size(q) q
      @info "K" size=size(k) k
      @info "V" size=size(v) v
      @info "Output" size=size(final) final
  end

  ╠═╡ =#

# ╔═╡ b9a4f0c7-09e2-489c-8297-3c2ae4e58d53
md"""
This is a test
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "512a5d8cba90deae6b891b95be568aec2818ed24"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "856ecd7cebb68e5fc87abecd2326ad59f0f911f3"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.43"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "7e35fca2bdfba44d797c53dfe63a51fabf39bfc0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.4.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "29bb0eb6f578a587a49da16564705968667f5fa8"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.2"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "a49f9342fc60c2a2aaa4e0934f06755464fcf438"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.6"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "3b704353e517a957323bd3ac70fa7b669b5f48d4"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.72.6"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EnzymeCore]]
git-tree-sha1 = "820f06722a87d9544f42679182eb0850690f9b45"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.17"
weakdeps = ["Adapt"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "5bfcd42851cf2f1b303f51525a54dc5e98d408a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.15.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Flux]]
deps = ["Adapt", "ChainRulesCore", "Compat", "EnzymeCore", "Functors", "LinearAlgebra", "MLCore", "MLDataDevices", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "Setfield", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "efa66783e2ad06bfd4c148cb34648e24c99f7626"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.16.7"

    [deps.Flux.extensions]
    FluxAMDGPUExt = "AMDGPU"
    FluxCUDAExt = "CUDA"
    FluxCUDAcuDNNExt = ["CUDA", "cuDNN"]
    FluxEnzymeExt = "Enzyme"
    FluxMPIExt = "MPI"
    FluxMPINCCLExt = ["CUDA", "MPI", "NCCL"]

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    NCCL = "3fe64909-d7a1-4096-9b7d-7a0f12cf0f6b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "b2977f86ed76484de6f29d5b36f2fa686f085487"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.3.1"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "60a0339f28a233601cb74468032b5c302d5067de"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "57e9ce6cf68d0abf5cb6b3b4abf9bedf05c939c0"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.15"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "b5a371fcd1d989d844a4354127365611ae1e305f"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.39"
weakdeps = ["EnzymeCore", "LinearAlgebra", "SparseArrays"]

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "73907695f35bc7ffd9f11f6c4f2ee8c1302084be"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.0.0"

[[deps.MLDataDevices]]
deps = ["Adapt", "Functors", "Preferences", "Random", "SciMLPublic"]
git-tree-sha1 = "d080e82120cc82114b4437780e03773d86d01c45"
uuid = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
version = "1.16.0"

    [deps.MLDataDevices.extensions]
    MLDataDevicesAMDGPUExt = "AMDGPU"
    MLDataDevicesCUDAExt = "CUDA"
    MLDataDevicesChainRulesCoreExt = "ChainRulesCore"
    MLDataDevicesChainRulesExt = "ChainRules"
    MLDataDevicesComponentArraysExt = "ComponentArrays"
    MLDataDevicesFillArraysExt = "FillArrays"
    MLDataDevicesGPUArraysExt = "GPUArrays"
    MLDataDevicesMLUtilsExt = "MLUtils"
    MLDataDevicesMetalExt = ["GPUArrays", "Metal"]
    MLDataDevicesOneHotArraysExt = "OneHotArrays"
    MLDataDevicesOpenCLExt = ["GPUArrays", "OpenCL"]
    MLDataDevicesReactantExt = "Reactant"
    MLDataDevicesRecursiveArrayToolsExt = "RecursiveArrayTools"
    MLDataDevicesReverseDiffExt = "ReverseDiff"
    MLDataDevicesSparseArraysExt = "SparseArrays"
    MLDataDevicesTrackerExt = "Tracker"
    MLDataDevicesZygoteExt = "Zygote"
    MLDataDevicescuDNNExt = ["CUDA", "cuDNN"]
    MLDataDevicesoneAPIExt = ["GPUArrays", "oneAPI"]

    [deps.MLDataDevices.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "MLCore", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "a772d8d1987433538a5c226f79393324b55f7846"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.8"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "09701dc1df4281fa9212b269a69210dfa81ee52a"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.32"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibMetalExt = "Metal"
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "bfe8e84c71972f77e775f75e6d8048ad3fdbe8bc"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.10"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "ConstructionBase", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "36b5d2b9dd06290cd65fcf5bdbc3a551ed133af5"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.4.7"

    [deps.Optimisers.extensions]
    OptimisersAdaptExt = ["Adapt"]
    OptimisersEnzymeCoreExt = "EnzymeCore"
    OptimisersReactantExt = "Reactant"

    [deps.Optimisers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "6ed167db158c7c1031abf3bd67f8e689c8bdf2b7"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.77"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "522f093a29b31a93e34eaea17ba055d850edea28"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.1"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "f0803bc1171e455a04124affa9c21bba5ac4db32"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.6"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLPublic]]
git-tree-sha1 = "0ba076dbdce87ba230fff48ca9bca62e1f345c9b"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.0.1"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "eee1b9ad8b29ef0d936e3ec9838c7ec089620308"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.16"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "be5733d4a2b03341bdcab91cea6caa7e31ced14b"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.9"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "a2c37d815bf00575332b7bd0389f771cb7987214"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.2"
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "SplittablesBase", "Tables"]
git-tree-sha1 = "4aa1fdf6c1da74661f6f5d3edfd96648321dade9"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.85"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

    [deps.UnsafeAtomics.weakdeps]
    LLVM = "929cbde3-209d-540e-8aea-75f648917ca0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "a29cbf3968d36022198bcc6f23fdfd70f7caf737"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.7.10"

    [deps.Zygote.extensions]
    ZygoteAtomExt = "Atom"
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Atom = "c52e3926-4ff0-5f6e-af25-54175e0327b1"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "434b3de333c75fc446aa0d19fc394edafd07ab08"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.7"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╟─5efe92c1-ba22-4b3b-a06f-c038b6ab9dde
# ╟─e36102b1-9ef1-4c7a-a61a-7cd66a757ec1
# ╟─1c2f8368-fc86-11f0-9ea7-c5cd0c983c87
# ╠═9bf0c59e-5cc0-40c2-8ece-43cdebb7f44a
# ╠═5a19623c-a055-4d4f-8671-aa1ff4734983
# ╠═19ce7080-77fc-4e86-9508-657b5211cfa0
# ╟─a562dfde-11d9-468a-ac38-228471873ca8
# ╠═127fe2ef-9dbb-435e-9b19-3ea049a2d58d
# ╟─8f72baaf-65a2-4a2f-99a7-d36dd283fedf
# ╠═3c781eb1-8ae9-4329-8978-eef444a0c411
# ╟─8d36c9ec-2d7d-44dc-9b04-b350545c6490
# ╠═904ff3d6-0831-435f-b0b7-ea194687c475
# ╟─1df3c443-4801-49b3-bc69-af5800427537
# ╠═a4019dbb-708e-4304-bea6-8804b6738259
# ╟─c4735111-c774-4f5a-8342-8128f5ff139d
# ╠═6fb42c0c-5d8a-41b5-8f00-daaef59e2bdd
# ╟─db43fda5-db51-49b0-9412-76d618587a09
# ╠═022dd229-826d-41d4-aef1-a5fd11cfd2cd
# ╟─ee93226c-2d62-41e6-ae4b-3129804a36da
# ╠═941a0f67-da14-434b-a34f-ab951f33a706
# ╟─138bf3b3-4d0e-4f1b-bcbb-c1479ad800a0
# ╠═40a0a544-256b-44ad-8272-d5cca9f751c4
# ╠═f6bfa719-022a-4aaa-b786-d9f020fa6e7d
# ╠═f7bf8738-899e-4f98-b18f-6d54576b64e4
# ╠═703d3823-54a7-4218-9977-8888a891396c
# ╠═28d10242-2d2c-45b0-88b3-ff58432d55bc
# ╟─7b2c4eed-dc18-442e-a5d9-0355eef143a7
# ╠═e5ccc157-574f-4f71-8288-05243441d804
# ╠═dcd6c8f2-6b64-456d-a24e-5f7b19370a09
# ╠═cb7dd025-19c1-4d87-a94f-2d19601ceebd
# ╠═2992e437-b934-40de-9f73-baa7bca4cf18
# ╠═b9a4f0c7-09e2-489c-8297-3c2ae4e58d53
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
