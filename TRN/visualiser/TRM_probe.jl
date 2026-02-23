module TRMProbe

using ..TRMMock  # Assuming TRMMock is available in the parent or same environment
using Flux
using NNlib
using Statistics


export ActivationLog, run_with_probe

"""
    ActivationLog

Stores the history of activations for a single input sequence.
- `y_history`: Vector of (Hidden, Seq, Batch) at each supervision step.
- `z_history`: Vector of (Hidden, Seq, Batch) at each inner latent step.
- `q_scores`: Vector of (2, Batch) Q-head predictions.
"""
struct ActivationLog

    y_history::Vector{Array{Float32, 3}}
    z_history::Vector{Array{Float32, 3}} 
    q_scores::Vector{Array{Float32, 2}}
    attn_history::Vector{Array{Float32, 3}} # (S, S, Heads*Batch) per step
end

"""
    forward_mha_probe(m::MHA, x, rope)

Returns (output, attention_weights)
"""
function forward_mha_probe(m::MHA, x, rope)
	D, S, B = size(x)
	hd, nh = m.hd, m.nh
	qkv = m.qkv(x)                           
	q = reshape(qkv[1:D, :, :],       hd, S, nh*B)
	k = reshape(qkv[D+1:2D, :, :],    hd, S, nh*B)
	v = reshape(qkv[2D+1:3D, :, :],   hd, S, nh*B)
	q, k = TRMMock.apply_rope(q, k, rope) # Explicit module reference
	sc = Float32(1/sqrt(hd))
	
	scores = NNlib.batched_mul(
				permutedims(q, (2,1,3)), k) .* sc     
	w = softmax(scores; dims=2)               # (S, S, nh*B)
	out = NNlib.batched_mul(v, permutedims(w, (2,1,3))) 
	proj = m.o_proj(reshape(out, D, S, B))
    
    return proj, w
end

"""
    forward_block_probe(b::TRMBlock, h, rope)

Returns (h_new, attn_weights)
"""
function forward_block_probe(b::TRMBlock, h, rope)
    attn_out, attn_weights = forward_mha_probe(b.attn, h, rope)
	h = TRMMock.rms_norm(h .+ attn_out; eps=b.eps)
	h = TRMMock.rms_norm(h .+ b.mlp(h); eps=b.eps)
    return h, attn_weights
end

"""
    forward_inner_probe(m::TRMInner, x_ids, y, z)

Like `forward_inner`, but returns intermediate `z` states and attention weights.
"""
function forward_inner_probe(m::TRMInner, x_ids, y, z)
    x_emb = m.embed(x_ids) .* Float32(sqrt(m.cfg.hidden_size))
    
    z_states = Array{Float32, 3}[]
    attn_weights = Array{Float32, 3}[]
    
    # Inner latent reasoning loop
    # ReasoningModule layers are TRMBlocks.
    # We need to iterate through them and probe each.
    
    # helper for probing reasoning module
    function probe_reasoning(h, injection, rope, layers)
        h = h .+ injection
        layer_attns = []
        for layer in layers
            h, w = forward_block_probe(layer, h, rope)
            push!(layer_attns, w)
        end
        # avg attention across layers? or keep all? 
        # For simplicity, let's just keep the last layer's attention or average them.
        # Let's average for now to save space in the visualizer, or just take the first one.
        # Taking the average of heads is common, but across layers is debatable.
        # Let's just take the first layer's attention for now as a representative.
        return h, layer_attns[1] 
    end

    for _ in 1:m.cfg.L_cycles
        # z = m.L_level(z, y .+ x_emb, m.rope) 
        # L_level IS the ReasoningModule
        z, attn = probe_reasoning(z, y .+ x_emb, m.rope, m.L_level.layers)
        
        push!(z_states, copy(z))
        push!(attn_weights, copy(attn))
    end
    
    # Answer update
    # y = m.L_level(y, z, m.rope)
    y, _ = probe_reasoning(y, z, m.rope, m.L_level.layers)
    
    logits   = m.lm_head(y)
    q_logits = m.q_head(y[:, 1, :])
    
    y, z, logits, q_logits, z_states, attn_weights
end

"""
    run_with_probe(model::TRM, x_ids; N_sup=nothing)

Runs the TRM model with deep supervision and records all states.
"""
function run_with_probe(model::TRM, x_ids; N_sup=nothing)
    cfg = model.inner.cfg
    limit = isnothing(N_sup) ? cfg.halt_max_steps : N_sup
    
    S, B = size(x_ids)
    y, z = TRMMock.init_yz(model, S, B)
    
    y_hist = [copy(y)] # Initial state
    z_hist = [copy(z)] 
    q_hist = Array{Float32,2}[]
    attn_hist = Array{Float32,3}[]
    
    for step in 1:limit
        T = cfg.H_cycles
        # We only really care about the final state of each supervision step for high-level viz?
        # But for deep understanding we want the inner loops. 
        # Let's record everything.
        
        
        q_logits = nothing
        for t in 1:T
            y, z, logits, q_logits, z_states, attn_weights = forward_inner_probe(model.inner, x_ids, y, z)
            
            append!(z_hist, z_states) 
            append!(attn_hist, attn_weights)
        end
        
        push!(y_hist, copy(y))
        push!(q_hist, q_logits)
    end
    
    # Pad attn_hist to match z_hist length if needed, or structured differently?
    # forward_inner_probe returns L_cycles states. T * L_cycles total.
    # z_hist has initial + T * L_cycles.
    # attn_hist has T * L_cycles.
    
    return ActivationLog(y_hist, z_hist, q_hist, attn_hist)
end

end # module
