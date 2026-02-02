### A Pluto.jl notebook ###
# v0.20.18

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 34438990-b233-11f0-28de-01899231a9ac
begin
    import Pkg
    Pkg.add([
        "Images",
        "ImageTransformations",
        "Interpolations",
        "FileIO",
        "Plots",
        "Interact",
        "PlutoUI",
        "WebIO"
    ])
    Pkg.build("WebIO")
end


# ╔═╡ 52b5a1b7-0bb3-4740-a857-e17083af2a54
begin
    using Images, ImageTransformations, Interpolations, FileIO
    using Plots, Interact, PlutoUI, WebIO
end


# ╔═╡ b02b8150-bd02-4eb6-a90c-dd8d120dfa92
begin
    function fisheye_transform(img; m=0.5)
        h, w = size(img)
        cx, cy = (w-1)/2, (h-1)/2
        max_r = sqrt(cx^2 + cy^2)

        function inverse_map(r_new)
            if abs(m) < 1e-8
                return r_new
            else
                disc = 1 + 2*m*r_new
                return (-1 + sqrt(disc)) / m
            end
        end

        itp = interpolate(img, BSpline(Linear()), OnGrid())
        img_itp = extrapolate(itp, 0)
        out = similar(img)

        for y in 1:h, x in 1:w
            dx, dy = x - cx, y - cy
            r_new = sqrt(dx^2 + dy^2) / max_r
            θ = atan(dy, dx)
            r = inverse_map(r_new) * max_r
            sx = cx + r * cos(θ)
            sy = cy + r * sin(θ)
            out[y, x] = img_itp(sy, sx)
        end

        return out
    end
end


# ╔═╡ 474a9b50-beca-4452-8eef-b011ddaf4502


# ╔═╡ ff0495cb-fc73-4555-9196-f34a93a772a8
begin
    # Create a reactive text box for image path input
    @bind file_path TextField(placeholder="Enter full path to your image file")
end


# ╔═╡ 2480b9f8-669c-4e3a-9e08-0ba702c9071b
begin
    if isempty(file_path)
        error("Please enter a valid image path above")
    end

    img = FileIO.load(file_path)
    println("Loaded image from: $file_path")
end


# ╔═╡ 2d4b5af5-e850-4ac6-befe-0db7654168e0
begin
   
    ms = [2, 5.0, 20.0, 30]

    # Prepare a layout: one row per m
    plt = plot(layout = (length(ms), 2), size=(800, 300*length(ms)))

    for (i, m) in enumerate(ms)
        fisheye_img = fisheye_transform(img; m=m)
        plot!(plt[i,1], img, title="Original", axis=nothing)
        plot!(plt[i,2], fisheye_img, title="Fisheye m=$m", axis=nothing)
    end

    plt  # Return the combined plot to Pluto
end


# ╔═╡ Cell order:
# ╠═34438990-b233-11f0-28de-01899231a9ac
# ╠═52b5a1b7-0bb3-4740-a857-e17083af2a54
# ╠═b02b8150-bd02-4eb6-a90c-dd8d120dfa92
# ╠═474a9b50-beca-4452-8eef-b011ddaf4502
# ╠═ff0495cb-fc73-4555-9196-f34a93a772a8
# ╠═2480b9f8-669c-4e3a-9e08-0ba702c9071b
# ╠═2d4b5af5-e850-4ac6-befe-0db7654168e0
