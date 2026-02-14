module TRMMock
# EXTRACTED FROM TRM_1.jl
# Removed Pluto-specific code and training loops to create a pure model definition file.

using Flux
using Flux: onehot, onehotbatch, logitcrossentropy
using LinearAlgebra
using Statistics
using Random
using Printf
using Zygote
using NNlib

export TRMConfig, SwiGLU, RoPE, MHA, TRMBlock, ReasoningModule, TRMInner, TRM,
       rms_norm, apply_rope, forward_inner, deep_recursion, init_yz

# --- Configuration ---
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
end

# --- Helper Functions ---
function rms_norm(x::AbstractArray; eps=1e-5f0)
	v = mean(x .^ 2; dims=1)
	x ./ sqrt.(v .+ eps)
end

# --- SwiGLU ---
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

# --- RoPE ---
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

# --- Multi-Head Attention ---
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

"""
function (m::MHA)(x, rope::RoPE)
	D, S, B = size(x)
	hd, nh = m.hd, m.nh
	qkv = m.qkv(x)                           # (3D, S, B)
	q = reshape(qkv[1:D, :, :],       hd, S, nh*B)
	k = reshape(qkv[D+1:2D, :, :],    hd, S, nh*B)
	v = reshape(qkv[2D+1:3D, :, :],   hd, S, nh*B)
	q, k = apply_rope(q, k, rope)
	sc = Float32(1/sqrt(hd))
	
	# scores = qTk / sqrt(d)
	scores = NNlib.batched_mul(
				permutedims(q, (2,1,3)), k) .* sc     # (S,S,nh*B)
	w = softmax(scores; dims=2)               # over keys
	out = NNlib.batched_mul(v, permutedims(w, (2,1,3)))  # (hd,S,nh*B)
	m.o_proj(reshape(out, D, S, B))
end
"""

# --- TRM Block ---
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

# --- Reasoning Module ---
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

# --- TRM Inner ---
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

# --- TRM Outer Wrapper ---
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

end # module
