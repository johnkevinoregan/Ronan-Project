module TRMVisualizer

using GLMakie
using Statistics
using ..TRMMock
using ..TRMProbe

export visualize_trm

"""
    visualize_trm(model::TRM, x_ids::Matrix{Int}; step_limit=nothing)

Launches an interactive Makie window to visualize the TRM dataflow.
"""
function visualize_trm(model::TRM, x_ids::Matrix{Int}; step_limit=nothing)
    # 1. Run Probe
    @info "Running model probe..."
    log = run_with_probe(model, x_ids; N_sup=step_limit)
    
    # 2. Setup Data
    S, B = size(x_ids)
    vocab_size = model.inner.cfg.vocab_size
    
    # Flatten history for slider
    # z_hist length = 1 + (N_sup * H_cycles * L_cycles)
    # y_hist length = 1 + N_sup
    # This mismatch makes synchronisation tricky.
    # Let's create a unified "Time Step" index.
    # We will visualise the 'z' state at the finest granularity.
    # We will replicate 'y' state to match 'z' resolution (hold previous value).
    
    total_steps = length(log.z_history)
    
    # Create Observables
    time_idx = Observable(1)
    
    # Helper to get data at step i
    # Since y updates much slower than z, we find the corresponding y index.
    # Ratio ≈ (H_cycles * L_cycles)
    cycles_per_sup = model.inner.cfg.H_cycles * model.inner.cfg.L_cycles
    
    # Y is updated every 'cycles_per_sup' steps? No, y is updated once per H_cycle actually?
    # In `run_with_probe`:
    #   Outer loop N_sup
    #     Inner loop T (H_cycles)
    #       forward_inner_probe calls:
    #         Init Z loop (1:L_cycles) -> z updates
    #         Last -> y update
    # So Y updates every (L_cycles) Z updates? No.
    # In `forward_inner`: z updates L times. Then y updates ONCE.
    # So for every L steps of z, we have 1 step of y.
    
    # Let's construct a mapping.
    # z_hist has all the intermediate Zs.
    # y_hist only has keyframes.
    
    # Actually, simpler: just visualize Z. Y is less interesting for "flow".
    # BUT, users want to see the answer `y` evolving.
    # We can just repeat the last known Y.
    
    # Attention Log:
    # attn_hist length is (N_sup * H_cycles * L_cycles) -- same as z_hist (minus initial).
    
    # Observables data
    z_data = @lift(log.z_history[$time_idx][:, :, 1]) # Take batch 1
    y_data = @lift begin
        # Map time_idx to appropriate Y.
        # Y is updated at the *end* of each forward_inner?
        # Actually `run_with_probe` only pushes to `y_hist` at the end of N_sup loop!
        # Wait, checking run_with_probe...
        # push!(y_hist, copy(y)) is AFTER the T-loop.
        # So Y is very coarse.
        # But `y` is passed into `forward_inner_probe`. It is constant during the inner loops.
        # So we can just use the Y from the beginning of the block.
        
        # Approximate mapping:
        # total_z = N_sup * T * L
        # total_y = N_sup
        # So for indices 1..(T*L), use y_hist[1].
        # For indices (T*L)+1 .. 2*(T*L), use y_hist[2].
        
        block_size = model.inner.cfg.H_cycles * model.inner.cfg.L_cycles
        y_idx = div($time_idx - 2, block_size) + 1
        y_idx = clamp(y_idx, 1, length(log.y_history))
        log.y_history[y_idx][:, :, 1]
    end
    
    attn_data = @lift begin
        if $time_idx <= 1 || $time_idx > length(log.attn_history) + 1
            zeros(Float32, S, S)
        else
            # attn_hist is 1-indexed relative to z_hist[2:end] effectively
            a = log.attn_history[$time_idx - 1]
            # Average over heads/batch
            mean(a, dims=3)[:, :, 1] 
        end
    end

    title_str = @lift "Step: $($time_idx) / $total_steps"

    # 3. Setup Layout
    fig = Figure(size = (1200, 800))
    
    # Top: Controls & Title
    Label(fig[1, :], title_str, fontsize=20, tellwidth=false)
    sl = Slider(fig[2, :], range = 1:total_steps, startvalue = 1)
    connect!(time_idx, sl.value)
    
    # Main Grid
    # Row 1: Z Heatmap | Y Heatmap
    # Row 2: Attention | Input Tokens
    
    ax_z = Axis(fig[3, 1], title="Latent Reasoning (z)", xlabel="Sequence", ylabel="Hidden Dim")
    hm_z = heatmap!(ax_z, z_data, colormap=:viridis)
    Colorbar(fig[3, 1][1, 2], hm_z, label="Activation")
    
    ax_y = Axis(fig[3, 2], title="Current Answer (y)", xlabel="Sequence", ylabel="Hidden Dim")
    hm_y = heatmap!(ax_y, y_data, colormap=:plasma)
    Colorbar(fig[3, 2][1, 2], hm_y)
    
    ax_attn = Axis(fig[4, 1], title="Self-Attention (Avg)", xlabel="Key Pos", ylabel="Query Pos", yreversed=true)
    hm_attn = heatmap!(ax_attn, attn_data, colormap=:blues, colorrange=(0, 1))
    Colorbar(fig[4, 1][1, 2], hm_attn)
    
    # Input Tokens display
    # We can plot them as text or a heatmap 1xS
    ax_tok = Axis(fig[4, 2], title="Input Sequence")
    # Just show text
    text_y = zeros(S)
    text_x = 1:S
    text!(ax_tok, string.(vec(x_ids[:, 1])), position = Point2f.(text_x, 0.5), align=(:center, :center), fontsize=16)
    xlims!(ax_tok, 0, S+1)
    hidespines!(ax_tok)
    hidedecorations!(ax_tok)
    
    display(fig)
    return fig
end

end # module
