### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ cebea14f-b104-42a8-9314-378a8b3bf210
begin
	using Pkg
	Pkg.activate(temp=true)
	Pkg.add(["Flux", "DataFrames", "CSV", "CairoMakie", "PlutoUI", "ProgressMeter",
			"Images"])	
	using Flux
	using Statistics
	using LinearAlgebra
	using Random
	using ProgressMeter
	using CairoMakie
	using PlutoUI
	using DataFrames
	using CSV
	using Markdown
	using Images.ImageCore
end

# ╔═╡ 3622e671-65c2-436d-b4cd-0b3ab8071ce7
HTML("""
<!-- the wrapper span -->
<div>
	<button id="myrestart" href="#">Restart</button>
	
	<script>
		const div = currentScript.parentElement
		const button = div.querySelector("button#myrestart")
		const cell= div.closest('pluto-cell')
		console.log(button);
		button.onclick = function() { restart_nb() };
		function restart_nb() {
			console.log("Restarting Notebook");
		        cell._internal_pluto_actions.send(                    
		            "restart_process",
                            {},
                            {
                                notebook_id: editor_state.notebook.notebook_id,
                            }
                        )
		};
	</script>
</div>
""")

# ╔═╡ e4e464bf-ea82-4520-bd5a-434cdceef303
md"""# Vision Grassmann Flow: 2D Spatial Adaptation

This notebook implements a 2D spatial variant of the Causal Grassmann Transformer for visual data (EMNIST)."""

# ╔═╡ 7b107f3e-b045-4145-bd4b-9b154b88a6a2
"""
Compute Plücker coordinates for the 2D subspace spanned by vectors u and v.
"""
function plucker_coordinates(u::AbstractVector{T}, v::AbstractVector{T}) where T
	r = length(u)
	M = u * v' - v * u'
	mask = [i < j for i in 1:r, j in 1:r]
	return M[mask]
end

# ╔═╡ fdd0cc22-3d8a-46c2-b5bf-fd3af16022eb
md"""## Spatial Grassmann Mixer

Adapts the Grassmann mixing to 2D spatial neighborhoods."""

# ╔═╡ 74da2341-8608-4cfc-ac1a-b6c8887d7575
begin
	struct SpatialGrassmannMixer
		W_red::Dense
		W_plu::Dense
		W_gate::Dense
		ln::LayerNorm
		d_model::Int
		r::Int
	windows::Vector{Tuple{Int, Int}}
	end

	function SpatialGrassmannMixer(d_model::Int, r::Int; 
								   windows=[(0,1), (1,0), (1,1), (-1,1)])
		n_plucker = r * (r - 1) ÷ 2
		SpatialGrassmannMixer(
			Dense(d_model => r),
			Dense(n_plucker => d_model),
			Dense(2 * d_model => d_model, Flux.sigmoid),
			LayerNorm(d_model),
			d_model,
			r,
			windows
		)
	end

	function _spatial_grassmann_col(m::SpatialGrassmannMixer, Z, t, b, 
									H_grid, W_grid, d, ::Type{T}) where T
		j = (t - 1) ÷ H_grid + 1
		i = (t - 1) % H_grid + 1
		
		valid_neighbors = filter(Δ -> (1 <= i + Δ[1] <= H_grid) && (1 <= j + Δ[2] <= W_grid), m.windows)
		
		if isempty(valid_neighbors)
			return zeros(T, d)
		else
			gs = map(valid_neighbors) do Δ
				n_i = i + Δ[1]
				n_j = j + Δ[2]
				n_idx = (n_j - 1) * H_grid + n_i
				
				z_t = Z[:, t, b]
				z_pair = Z[:, n_idx, b]
				
				p = plucker_coordinates(z_t, z_pair)
				p_norm = max(norm(p), T(1e-8))
				m.W_plu(p ./ p_norm)
			end
			return reduce(.+, gs) ./ T(length(gs))
		end
	end

	function (m::SpatialGrassmannMixer)(X::AbstractArray{T, 3}) where T
		d, seq_len, B = size(X)
		H_grid = Int(sqrt(seq_len))
		W_grid = H_grid
		
		X_flat = reshape(X, d, seq_len * B)
		Z = m.W_red(X_flat)
		Z = reshape(Z, m.r, seq_len, B)
		
		G_cols = [_spatial_grassmann_col(m, Z, t, b, H_grid, W_grid, d, T) 
				  for t in 1:seq_len, b in 1:B]
		G = reshape(reduce(hcat, vec(G_cols)), d, seq_len, B)
		
		G_flat = reshape(G, d, seq_len * B)
		U = vcat(X_flat, G_flat)
		
		α = m.W_gate(U)
		X_mixed = α .* X_flat .+ (one(T) .- α) .* G_flat
		
		X_out = m.ln(X_mixed)
		return reshape(X_out, d, seq_len, B)
	end
end

# ╔═╡ 9e94a37f-f2cc-4d24-88d8-51b83a92a20c
Flux.@layer SpatialGrassmannMixer

# ╔═╡ fe2eb33c-256c-4d3b-8927-157f04cd1631
md"""## Vision Blocks and Architecture"""

# ╔═╡ a976ca65-de5f-496f-937c-9ffb6129d48a
begin
	struct FeedForward
		fc1::Dense
		fc2::Dense
		ln::LayerNorm
		dropout::Dropout
	end

	function FeedForward(d_model::Int; d_ff=nothing, dropout_rate=0.1)
		d_ff = isnothing(d_ff) ? 4 * d_model : d_ff
		FeedForward(
			Dense(d_model => d_ff, Flux.gelu),
			Dense(d_ff => d_model),
			LayerNorm(d_model),
			Dropout(dropout_rate)
		)
	end
	
	function (ff::FeedForward)(x)
		d, L, B = size(x)
		x_flat = reshape(x, d, L * B)
		h = ff.fc1(x_flat)
		h = ff.fc2(h)
		h = ff.dropout(h)
		out = ff.ln(x_flat .+ h)
		return reshape(out, d, L, B)
	end
end

# ╔═╡ 39073329-fea1-4fb5-80ab-93666dbcca1a
Flux.@layer FeedForward

# ╔═╡ 8a9fbadd-4980-4df4-8b78-2cda45477da0
begin
	struct VisionGrassmannBlock
		mixer::SpatialGrassmannMixer
		ffn::FeedForward
	end

	function VisionGrassmannBlock(d_model::Int, r::Int; windows=[(0,1), (1,0),
																 (1,1), (-1,1)], d_ff=nothing, dropout_rate=0.1)
		VisionGrassmannBlock(
			SpatialGrassmannMixer(d_model, r; windows=windows),
			FeedForward(d_model; d_ff=d_ff, dropout_rate=dropout_rate)
		)
	end
	
	function (block::VisionGrassmannBlock)(x)
		h = block.mixer(x)
		return block.ffn(h)
	end
end

# ╔═╡ b3018993-9f7e-4ad5-a8a9-edd2d1b6c2ee
Flux.@layer VisionGrassmannBlock

# ╔═╡ fe1dfa25-bc57-407e-9af5-a064b858d918
begin
	struct VisionGrassmannClassifier
		patch_proj::Dense
		pos_embedding::AbstractMatrix
		blocks::Vector{VisionGrassmannBlock}
		classifier::Chain
		d_model::Int
	end

	function VisionGrassmannClassifier(patch_size::Int, d_model::Int, 
								   n_layers::Int, r::Int, n_classes::Int;
		seq_len=49, windows=[(0,1), (1,0), (1,1), (-1,1), (0,-1), (-1,0), 
							 (-1,-1), (1,-1)])
		patch_dim = patch_size * patch_size
		pos_emb = randn(Float32, d_model, seq_len) .* 0.02f0
		
		blocks = [VisionGrassmannBlock(d_model, r; windows=windows, 
									   dropout_rate=0.1) for _ in 1:n_layers]
		
		VisionGrassmannClassifier(
			Dense(patch_dim => d_model),
			pos_emb,
			blocks,
			Chain(
				LayerNorm(d_model),
				Dense(d_model => d_model ÷ 2, Flux.relu),
				Dropout(0.1),
				Dense(d_model ÷ 2 => n_classes)
			),
			d_model
		)
	end
	
	function (model::VisionGrassmannClassifier)(x::AbstractMatrix{T}) where T
		# x: (patch_dim, seq_len * batch_size)
		patch_dim, N = size(x)
		seq_len = size(model.pos_embedding, 2)
		B = N ÷ seq_len
		
		h_flat = model.patch_proj(x) # (d_model, seq_len * B)
		h = reshape(h_flat, model.d_model, seq_len, B)
		
		pos = @view model.pos_embedding[:, 1:seq_len]
		h = h .+ reshape(pos, model.d_model, seq_len, 1)
		
		for block in model.blocks
			h = block(h)
		end
		
		h_pooled = dropdims(mean(h, dims=2), dims=2)
		return model.classifier(h_pooled)
	end
end

# ╔═╡ 0db52e1c-f8b0-43de-90c5-7c48c6e409b2
Flux.@layer VisionGrassmannClassifier

# ╔═╡ 767e5822-0481-41fd-b739-af34a69d4e74
md"""## Data Loading, Training, and Visualizations"""

# ╔═╡ b881f626-92f9-406e-84d0-a518ab8d6eb1
begin
	function load_emnist_data(csv_path, max_samples=1000)
		df = CSV.read(csv_path, DataFrame; limit=max_samples, header=false)
		
		# First column is label
		Y = Vector{Int}(df[!, 1])
		
		# Remaining columns are pixels
		X_pixels = Matrix{Float32}(df[!, 2:end]) ./ 255.0f0
		
		return X_pixels, Y
	end
	
	function extract_patches(X_pixels, img_size=28, patch_size=4)
		N = size(X_pixels, 1)
		grid_size = img_size ÷ patch_size
		seq_len = grid_size * grid_size
		patch_dim = patch_size * patch_size
		
		# Output shape needed: (patch_dim, seq_len * N)
		patches = zeros(Float32, patch_dim, seq_len * N)
		
		for n in 1:N
			# EMNIST images need transposing due to how they are flattened
			img = transpose(reshape(X_pixels[n, :], img_size, img_size))
			
			for i in 1:grid_size
				for j in 1:grid_size
					y_start = (i - 1) * patch_size + 1
					y_end = i * patch_size
					x_start = (j - 1) * patch_size + 1
					x_end = j * patch_size
					
					patch = img[y_start:y_end, x_start:x_end]
					
					t = (j - 1) * grid_size + i
					idx = (n - 1) * seq_len + t
					patches[:, idx] = vec(patch)
				end
			end
		end		
		return patches
	end
end

# ╔═╡ 18a34e56-2f4a-4d75-ace3-95c430797a23
function train_vision_classifier!(model, X_train, Y_train, X_val, Y_val; 
								  epochs=10, batch_size=32, lr=1e-3, seq_len=49)
	opt_state = Flux.setup(Adam(lr), model)
	
	n_train = length(Y_train)
	n_batches = n_train ÷ batch_size
	
	train_losses = Float32[]
	val_accs = Float32[]
	
	for epoch in 1:epochs
		perm = randperm(n_train)
		epoch_loss = 0f0
		
		for i in 1:n_batches
			idx_start = (i - 1) * batch_size + 1
			idx_end = i * batch_size
			batch_indices = perm[idx_start:idx_end]
			
			Y_batch = Y_train[batch_indices]
			
			# Construct X_batch of size (patch_dim, seq_len * batch_size)
			X_batch = zeros(Float32, size(X_train, 1), seq_len * batch_size)
			for (b_idx, orig_idx) in enumerate(batch_indices)
				src_start = (orig_idx - 1) * seq_len + 1
				src_end = orig_idx * seq_len
				dst_start = (b_idx - 1) * seq_len + 1
				dst_end = b_idx * seq_len
				X_batch[:, dst_start:dst_end] = X_train[:, src_start:src_end]
			end
			
			loss, grads = Flux.withgradient(model) do m
				logits = m(X_batch)
				Flux.logitcrossentropy(logits, Flux.onehotbatch(Y_batch, 0:46))
			end
			
			epoch_loss += loss
			Flux.update!(opt_state, model, grads[1])
		end
		
		push!(train_losses, epoch_loss / n_batches)
		
		# Simple validation logic for small sets
		n_val = length(Y_val)
		val_preds = []
		for i in 1:n_val
			src_start = (i - 1) * seq_len + 1
			src_end = i * seq_len
			X_v = X_val[:, src_start:src_end]
			logits = model(X_v)
			push!(val_preds, Flux.onecold(logits, 0:46)[1])
		end
		val_acc = mean(val_preds .== Y_val)
		push!(val_accs, val_acc)
		
		println("Epoch $epoch: Loss = $(round(train_losses[end], digits=4)), Val Acc = $(round(val_acc * 100, digits=2))%")
	end
	
	return train_losses, val_accs
end

# ╔═╡ 19ec161c-11b9-4988-b5b2-8d60faa15c61
md"""## Experiment Execution"""

# ╔═╡ df738f20-72a1-4714-9dd6-5c5bff188f37
begin
	Random.seed!(42)
	
	# Hyperparameters
	d_model = 64
	n_layers = 2
	r = 16
	n_classes = 47 # EMNIST balanced has 47 classes
	patch_size = 4 # 28x28 image -> 7x7 grid = 49 patches
	seq_len = 49
	
	println("Loading Data...")
	X_train_raw, Y_train = load_emnist_data("/Users/rgreilly/Documents/GrassmanFlow/EMNIST/emnist-balanced-train.csv", 5000)
	X_val_raw, Y_val = load_emnist_data("/Users/rgreilly/Documents/GrassmanFlow/EMNIST/emnist-balanced-test.csv", 1000)
	
	println("Extracting Patches...")
	X_train = extract_patches(X_train_raw)
	X_val = extract_patches(X_val_raw)
	
	println("Data Ready. Train samples: $(length(Y_train)), Val samples: $(length(Y_val))")
end

# ╔═╡ 1ed99ae1-e237-46b8-85ca-613ad16e2590
begin
	vision_model = VisionGrassmannClassifier(patch_size, d_model, 
											 n_layers, r, n_classes)
	flat_params, _ = Flux.destructure(vision_model)
	println("Model parameters: ", length(flat_params))
end

# ╔═╡ deb7edef-e877-440b-b255-e06145ea7e34
begin
	println("\nTraining Vision Grassmann Classifier...")
	train_losses, val_accs = train_vision_classifier!(
			vision_model, X_train, Y_train, X_val, Y_val;
			epochs=50, batch_size=32, lr=1e-3, seq_len=49
		)
end

# ╔═╡ 1e126939-bc60-4306-9b8c-b2ec43db9371
begin
	f = Figure(size = (900, 350))
	ax1 = Axis(f[1, 1], title = "Training Loss", xlabel = "Epoch", ylabel = "Loss")
	lines!(ax1, train_losses, label = "Training Loss", linewidth = 2)
	
	ax2 = Axis(f[1, 2], title = "Validation Accuracy", xlabel = "Epoch", ylabel = "Accuracy (%)")
	lines!(ax2, val_accs .* 100, label = "Validation Accuracy", linewidth = 2, color = :green)
	f
end

# ╔═╡ b8613fea-670d-4617-a077-379c3274c8c2
md"""## Visualisations: Characters and Patches"""

# ╔═╡ dc765b81-5f93-4737-820f-7ea43a0dacc3
begin
	# Mapping from EMNIST balanced dataset class index to actual character
	# Class 0-9: digits '0'-'9'
	# Class 10-35: uppercase 'A'-'Z'
	# Class 36-46: selected lowercase
	function get_emnist_char(class_idx)
		if 0 <= class_idx <= 9
			return Char('0' + class_idx)
		elseif 10 <= class_idx <= 35
			return Char('A' + class_idx - 10)
		else
			# Mapping for remaining lowercase letters based on the EMNIST balanced spec
			# 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
			ext_map = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
			return ext_map[class_idx - 35]
		end
	end
end

# ╔═╡ 6930140c-d9a8-4377-bc98-cc94863c3ea6
begin
	function plot_sample_characters(X_raw, Y_raw; num_samples=5)
		f = Figure(size=(800,200))
		for i in 1:num_samples
			img = reshape(X_raw[i, :], 28, 28)
			# EMNIST images typically need a transpose to be viewed correctly
			img = transpose(img)
			ax = Axis(f[1, i], 
					  title="Class: $(Y_raw[i]) ('$(get_emnist_char(Y_raw[i]))')",
					  titlesize=10, yreversed=true)
			hidedecorations!(ax)
			heatmap!(ax, img)
		end
		return f
	end

	p_characters = plot_sample_characters(X_val_raw, Y_val; num_samples=5)
end

# ╔═╡ 7d3b8d0f-9c1c-46b5-8396-56c084ace211
begin
	function plot_patches_for_sample(X_patches, sample_idx=1; grid_size=7)
		# X_patches is (patch_dim, seq_len * N)
		seq_len = grid_size * grid_size
		patch_size = Int(sqrt(size(X_patches, 1)))
		
		src_start = (sample_idx - 1) * seq_len + 1
		src_end = sample_idx * seq_len
		patches = X_patches[:, src_start:src_end]
		
		f = Figure(size=(400, 400))
		for j in 1:grid_size
			for i in 1:grid_size
				t = (j - 1) * grid_size + i
				patch_img = reshape(patches[:, t], patch_size, patch_size)
				ax = Axis(f[j, i], yreversed=true)
				hidedecorations!(ax)
				heatmap!(ax, patch_img)
				
			end
		end
		Label(f[0, :], "7x7 grid of 4x4 patches", fontsize=20)
		return f
	end
	
	p_patches = plot_patches_for_sample(X_val, 1)
end

# ╔═╡ fdfc8e84-c1ee-487e-bd79-f0e734f06841
md"""## Visualisations: Plücker Norms & Internal Geometry"""

# ╔═╡ 1c746143-8a1e-4d09-8961-4c36a17e0be9
begin
function analyze_plucker_norms(model, X_val, Y_val; sample1_idx=1, sample2_idx=2)
		# Extract specific samples
		seq_len = 49
		src_start1 = (sample1_idx - 1) * seq_len + 1
		src_end1 = sample1_idx * seq_len
		x1 = X_val[:, src_start1:src_end1]
		
		src_start2 = (sample2_idx - 1) * seq_len + 1
		src_end2 = sample2_idx * seq_len
		x2 = X_val[:, src_start2:src_end2]
		
		# Propagate to the input of the first Grassmann mixer
		function get_z(x)
			h_flat = model.patch_proj(x)
			h = reshape(h_flat, model.d_model, seq_len, 1)
			pos = @view model.pos_embedding[:, 1:seq_len]
			h = h .+ reshape(pos, model.d_model, seq_len, 1)
			
			h_flat_mixer = reshape(h, model.d_model, seq_len * 1)
			
			# Getting the reduced space components from block 1 mixer
			Z = model.blocks[1].mixer.W_red(h_flat_mixer)
			return reshape(Z, model.blocks[1].mixer.r, seq_len)
		end
		
		z1 = get_z(x1)
		z2 = get_z(x2)
		
		# Define interesting comparisons 
		# 1. Compare patch pairs inside same image (spatial smoothness)
		r = size(z1, 1)
		H_grid = 7
		
		intra_norms = Float32[]
		for i in 1:(H_grid-1)
			for j in 1:(H_grid-1)
				t = (j - 1) * H_grid + i
				t_right = (j - 1) * H_grid + (i + 1)
				
				p = plucker_coordinates(z1[:, t], z1[:, t_right])
				push!(intra_norms, norm(p))
			end
		end
		
		# 2. Compare patch pairs across different images (same nominal location)
		cross_norms = Float32[]
		for t in 1:seq_len
			p = plucker_coordinates(z1[:, t], z2[:, t])
			push!(cross_norms, norm(p))
		end
		
		# 3. Compare randomly selected patches
		rand_norms = Float32[]
		for _ in 1:50
			t1 = rand(1:seq_len)
			t2 = rand(1:seq_len)
			p = plucker_coordinates(z1[:, t1], z2[:, t2])
			push!(rand_norms, norm(p))
		end
		
		c1, c2 = get_emnist_char(Y_val[sample1_idx]), get_emnist_char(Y_val[sample2_idx])

		f = Figure(size=(400, 500))
		ax = Axis(f[1,:], xlabel="||p||", ylabel="frequency")		
		hist!(ax, intra_norms, alpha=0.5, 
			  label="intra-image adjacent ($(c1))", bins=15)
		hist!(ax, cross_norms, alpha=0.5, 
				   label="cross-image aligned ($(c1) vs $(c2))", bins=15)
		hist!(ax, rand_norms, alpha=0.5, label="random pairs", bins=15)
		axislegend(ax)

		Label(f[0, :], "Plücker Norm Distributions", fontsize=20)

		return f
	end
	
	p_norms = analyze_plucker_norms(vision_model, X_val, Y_val, sample1_idx=1, sample2_idx=2)
end

# ╔═╡ 71b3d54a-021e-4f10-a82c-30a942914b18
begin
    # To visualise internal states across depth
    function visualize_internal_activations(model, X_val, sample_idx=1)
        seq_len = 49
        src_start = (sample_idx - 1) * seq_len + 1
        src_end = sample_idx * seq_len
        x = X_val[:, src_start:src_end]
        
        h_flat = model.patch_proj(x)
        h = reshape(h_flat, model.d_model, seq_len, 1)
        pos = @view model.pos_embedding[:, 1:seq_len]
        h = h .+ reshape(pos, model.d_model, seq_len, 1)
        
        activations = [copy(reshape(h, model.d_model, seq_len))]
        
        for block in model.blocks
            h = block(h)
            push!(activations, copy(reshape(h, model.d_model, seq_len)))
        end
        
        # Calculate L2 norm of the hidden traits for each patch
        norms = [ [norm(act[:, t]) for t in 1:seq_len] for act in activations ]
        
        f = Figure(size=(900, 250))
        titles = ["Input Embeddings", "After Block 1", "After Block 2"]
        
        for (i, n) in enumerate(norms)
            grid_n = reshape(n, 7, 7)
            # Normalise to 0-1 for consistent display
            grid_n = (grid_n .- minimum(grid_n)) ./ 
				(maximum(grid_n) - minimum(grid_n) + 1e-6)
            
            ax = Axis(f[1, i], title=titles[i])
            hidedecorations!(ax)
            hidespines!(ax)
            
            heatmap!(ax, grid_n, colormap=:viridis)
            ax.yreversed = true
        end
        
        Label(f[0, :], "Activation Energy (Norms) Across Layers", fontsize=20)
        return f
    end

    p_activations = visualize_internal_activations(vision_model, X_val, 1)
end

# ╔═╡ Cell order:
# ╟─3622e671-65c2-436d-b4cd-0b3ab8071ce7
# ╠═cebea14f-b104-42a8-9314-378a8b3bf210
# ╟─e4e464bf-ea82-4520-bd5a-434cdceef303
# ╠═7b107f3e-b045-4145-bd4b-9b154b88a6a2
# ╟─fdd0cc22-3d8a-46c2-b5bf-fd3af16022eb
# ╠═74da2341-8608-4cfc-ac1a-b6c8887d7575
# ╠═9e94a37f-f2cc-4d24-88d8-51b83a92a20c
# ╟─fe2eb33c-256c-4d3b-8927-157f04cd1631
# ╠═a976ca65-de5f-496f-937c-9ffb6129d48a
# ╠═39073329-fea1-4fb5-80ab-93666dbcca1a
# ╠═8a9fbadd-4980-4df4-8b78-2cda45477da0
# ╠═b3018993-9f7e-4ad5-a8a9-edd2d1b6c2ee
# ╠═fe1dfa25-bc57-407e-9af5-a064b858d918
# ╠═0db52e1c-f8b0-43de-90c5-7c48c6e409b2
# ╟─767e5822-0481-41fd-b739-af34a69d4e74
# ╠═b881f626-92f9-406e-84d0-a518ab8d6eb1
# ╠═18a34e56-2f4a-4d75-ace3-95c430797a23
# ╟─19ec161c-11b9-4988-b5b2-8d60faa15c61
# ╠═df738f20-72a1-4714-9dd6-5c5bff188f37
# ╠═1ed99ae1-e237-46b8-85ca-613ad16e2590
# ╠═deb7edef-e877-440b-b255-e06145ea7e34
# ╠═1e126939-bc60-4306-9b8c-b2ec43db9371
# ╟─b8613fea-670d-4617-a077-379c3274c8c2
# ╠═dc765b81-5f93-4737-820f-7ea43a0dacc3
# ╠═6930140c-d9a8-4377-bc98-cc94863c3ea6
# ╠═7d3b8d0f-9c1c-46b5-8396-56c084ace211
# ╟─fdfc8e84-c1ee-487e-bd79-f0e734f06841
# ╠═1c746143-8a1e-4d09-8961-4c36a17e0be9
# ╠═71b3d54a-021e-4f10-a82c-30a942914b18
