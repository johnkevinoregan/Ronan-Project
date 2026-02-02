### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ db7f3b5e-7916-45e3-894c-032f44dbe303
using Random

# ╔═╡ ebefb095-6f96-4bcd-a8ad-e34c6b7649b7
begin

@inline color_index(x::Int, y::Int) = (x & 1) + 2*(y & 1)
	
struct SpatialBumpMapUnique
    centers::Matrix{Float32}  # 2×M_bump
    R::Float32
    width::Int
    height::Int
end

function SpatialBumpMapUnique(M_total::Int, k_total::Int,
                              width::Int, height::Int;
                              rng=Random.GLOBAL_RNG)
    @assert M_total > 4 "Need at least 5 dims (4 for color + ≥1 for bumps)"
    @assert k_total > 1 "Need at least 2 active bits"

    M_bump = M_total - 4
    k_bump = k_total - 1

    A = width * height
    R = sqrt(k_bump * A / (M_bump * π))

    centers = rand(rng, Float32, 2, M_bump)
    centers[1, :] .*= (width  - 1)
    centers[2, :] .*= (height - 1)

    return SpatialBumpMapUnique(centers, Float32(R), width, height)
end

function (f::SpatialBumpMapUnique)(X::AbstractMatrix)
    @assert size(X, 1) == 2 "X must be 2×N with rows (x,y)"

    M_bump  = size(f.centers, 2)
    M_total = M_bump + 4
    N       = size(X, 2)
    R2      = f.R * f.R

    out = falses(M_total, N)

    @inbounds for i in 1:N
        x = Int(round(X[1, i]))
        y = Int(round(X[2, i]))

        # bump part
        for j in 1:M_bump
            cx = f.centers[1, j]
            cy = f.centers[2, j]
            dx = Float32(x) - cx
            dy = Float32(y) - cy
            if dx*dx + dy*dy <= R2
                out[j, i] = true
            end
        end

        # 4-color tag
        c = color_index(x, y)               # 0..3
        out[M_bump + 1 + c, i] = true
    end

    return out          # Bool M_total×N
end
end

# ╔═╡ b9b5ab2e-f268-4f2c-816c-433fe1df401b
begin
width  = 512
height = 512

M_total = 1024   # output dimension
k_total = 16     # approx # active bits

embed = SpatialBumpMapUnique(M_total, k_total, width, height)
end

# ╔═╡ c36553ba-9664-44ec-9642-08cd43395df1
X = [10  11  200  300;
     10  11  210  310]     # size = (2, 4)


# ╔═╡ e5cfaf12-b0c8-4299-857a-4761cf5a8879
Z_bool = embed(X)


# ╔═╡ feaa99b6-7ed2-4dba-b4d5-3ad3970c469c
size(Z_bool)


# ╔═╡ f3bf86a2-5625-428e-b13d-9944f1bef13f
for i in 1:20
    println(Z_bool[i, 1:4])
end


# ╔═╡ 83e5f1d7-6f22-4a1d-86be-f9742d0c5e46
begin
overlap_adj = sum(Z_bool[:,1] .& Z_bool[:,2]) / sum(Z_bool[:,1] .| Z_bool[:,2])
println("Overlap fraction between (10,10) and (11,11): ", overlap_adj)
end

# ╔═╡ 5ad5b274-6eeb-47e0-ba62-04a1ddc728d0
# ╠═╡ disabled = true
#=╠═╡
begin
	overlap_far = sum(Z_bool[:,1] .& Z_bool[:,4]) / sum(Z_bool[:,1] .| Z_bool[:,4])
	println("Overlap far apart: ", overlap_far)
end
  ╠═╡ =#

# ╔═╡ 7090db8b-865e-41a7-97da-652123af0303
begin
overlap_far = sum(Z_bool[:,1] .& Z_bool[:,4]) / sum(Z_bool[:,1] .| Z_bool[:,4])
println("Overlap far apart: ", overlap_far)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.2"
manifest_format = "2.0"
project_hash = "9e64cf17f9522d20edabe6f2b4ec85252943fcae"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"
"""

# ╔═╡ Cell order:
# ╠═db7f3b5e-7916-45e3-894c-032f44dbe303
# ╠═ebefb095-6f96-4bcd-a8ad-e34c6b7649b7
# ╠═b9b5ab2e-f268-4f2c-816c-433fe1df401b
# ╠═c36553ba-9664-44ec-9642-08cd43395df1
# ╠═e5cfaf12-b0c8-4299-857a-4761cf5a8879
# ╠═feaa99b6-7ed2-4dba-b4d5-3ad3970c469c
# ╠═f3bf86a2-5625-428e-b13d-9944f1bef13f
# ╠═5ad5b274-6eeb-47e0-ba62-04a1ddc728d0
# ╠═83e5f1d7-6f22-4a1d-86be-f9742d0c5e46
# ╠═7090db8b-865e-41a7-97da-652123af0303
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
