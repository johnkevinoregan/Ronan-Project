using Flux
using GLMakie
using Random

include("TRM_core.jl")
include("TRM_probe.jl")
include("TRMVisualizer.jl")

using .TRMMock
using .TRMProbe
using .TRMVisualizer

println("Initializing TRM Model...")

# 1. Define Configuration
# A slightly larger model than the test one, for better viz
V, S, B, D = 3, 5, 1, 16		#KOR was 15, 16, 1, 64
cfg = TRMConfig(
    vocab_size=V, seq_len=S, hidden_size=D,
    num_heads=2, H_cycles=3, L_cycles=6, L_layers=2,		#KOR was 4, 3, 6, 2
    halt_max_steps=16
)

model = TRM(cfg)

# 2. Generate Random Input
# In a real scenario, you might want to load a specific problem instance
x_ids = rand(1:V, S, B)

println("Launching Visualization...")
println("Step through the model using the slider at the top.")

# 3. Visualize
fig = visualize_trm(model, x_ids)
wait(display(fig))
