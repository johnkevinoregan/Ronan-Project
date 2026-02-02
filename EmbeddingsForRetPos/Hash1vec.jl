### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ f7d3c4ec-7ddd-4ece-8ebd-6d98a791644e
using SparseArrays

# ╔═╡ cba33458-d0c9-44b0-98b8-393bcf809b24


"""
    hash_k_sparse_exact(vec; M, k=2, seed=0x12345678)

vec :: AbstractVector or Tuple of length 2, containing integers [x, y].

Returns an M-dimensional sparse vector with exactly k ones.
"""
function hash_k_sparse_exact(vec::AbstractVector; M::Integer,
                             k::Integer=2, seed::Unsigned=0x12345678)

    @assert length(vec) == 2 "Input vector must have length 2"
    x, y = vec[1], vec[2]

    @assert M ≥ k "M must be at least k"

    idxset = Set{Int}()
    hseed = UInt(seed)
    i = 1

    while length(idxset) < k
        h = hash((x, y, i), hseed)
        j = Int(rem(h, UInt(M)) + 1)
        push!(idxset, j)
        i += 1
    end

    idxs = collect(idxset)
    vals = ones(Int8, length(idxs))

    return SparseVector{Int8,Int}(M, idxs, vals)
end


# ╔═╡ 89c5eea7-7ebe-468b-b79c-6e1cbb5f98fd
v = hash_k_sparse_exact([10, 3]; M=1000, k=4, seed=UInt(0x12345678))

# ╔═╡ 4ccf105f-60f9-4c39-8520-9ff49d3c9d7a
println(v)

# ╔═╡ 1f968e14-e8b5-471b-ae1d-348a7331a12c
println(nnz(v))  # should be 4

# ╔═╡ 221eafd4-c67d-431c-bb5d-2495d81b50ed
println(v)

# ╔═╡ e6e9a084-af71-47ff-88d1-de647ef765c8
#
# Convert sparse → full 0/1 vector of length M
#
dense(v::SparseVector) = Int.(collect(v))  # convert to dense array of Int

# ╔═╡ c3c00c67-2a6c-4b66-aee8-ffdf61278f87
# Print sparse vectors for all (x,y) in 0..4 × 0..4
#
M = 20

# ╔═╡ b444fb76-ede0-4c1a-a9de-cd09bf433d4f
K = 4

# ╔═╡ 89278573-a5c9-4a90-ac21-84092c0808ff

for x in 0:0.5:2
    for y in 0:4
        sv = hash_k_sparse_exact([x, y]; M=M, k=K)
        dv = dense(sv)
        println("(x,y)=($x,$y) → ", dv)
    end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.2"
manifest_format = "2.0"
project_hash = "60cf12fdb22b35802055516ae033d7e0a6a35dae"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"
"""

# ╔═╡ Cell order:
# ╠═f7d3c4ec-7ddd-4ece-8ebd-6d98a791644e
# ╠═89c5eea7-7ebe-468b-b79c-6e1cbb5f98fd
# ╠═cba33458-d0c9-44b0-98b8-393bcf809b24
# ╠═4ccf105f-60f9-4c39-8520-9ff49d3c9d7a
# ╠═1f968e14-e8b5-471b-ae1d-348a7331a12c
# ╠═221eafd4-c67d-431c-bb5d-2495d81b50ed
# ╠═e6e9a084-af71-47ff-88d1-de647ef765c8
# ╠═c3c00c67-2a6c-4b66-aee8-ffdf61278f87
# ╠═b444fb76-ede0-4c1a-a9de-cd09bf433d4f
# ╠═89278573-a5c9-4a90-ac21-84092c0808ff
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
