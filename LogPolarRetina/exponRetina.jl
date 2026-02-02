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

# ╔═╡ d3ecb7f4-b248-11f0-1956-f1af898547ac
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


# ╔═╡ 659946da-45d6-48b5-9747-3aeaf142b28f
begin
    using Images, ImageTransformations, Interpolations, FileIO
    using Plots, Interact, PlutoUI, WebIO
end


# ╔═╡ 61df4f3d-602d-4537-a950-399aedb86e25
begin
function expon_transform(img; s0=0.00277453, rate=0.7666667, vis_field_radius_deg=0.5)
	M=0.3 #number of mms of retina corresponding to 1°
	image_radius_mm = vis_field_radius_deg*M
    h, w = size(img)[1:2]
    cx, cy = (w-1)/2, (h-1)/2
    image_radius_pix = sqrt(cx^2 + cy^2)
	

    # Prepare interpolation
    itp = interpolate(img, BSpline(Linear()), OnGrid())
    img_itp = extrapolate(itp, 0)
    out = similar(img)

    for y in 1:h, x in 1:w
        dx, dy = x - cx, y - cy
        r_pix = sqrt(dx^2 + dy^2) / image_radius_pix
        θ = atan(dy, dx)

        # log scaling
		r_mm = image_radius_mm * (r_pix/image_radius_pix)
        r_scaled_mm = (s0/rate)*(exp(rate*r_mm)-1)
		r_scaled_pix = r_scaled_mm * (image_radius_pix / image_radius_mm)

        # Map back to image coordinates
        sx = cx + r_scaled_pix*cos(θ)
        sy = cy + r_scaled_pix*sin(θ)

        # Handle RGB and grayscale
        if ndims(img) == 3
            out[y,x,:] = img_itp(sy,sx,:)
        else
            out[y,x] = img_itp(sy,sx)
        end
    end

    return out
end

end




# ╔═╡ 03737d26-4a70-4612-a436-69819db34416
begin
    @bind file_path TextField(placeholder="Enter full path to your image file")
end


# ╔═╡ b87f0800-f4ea-4ece-bdea-77771a15bda7
begin
    if isempty(file_path)
        error("Please enter a valid image path above")
    end
	# doesnt seem to work: filepath = "/Users/kevinoregan/Desktop/Dropbox/Temporary/Grid6.jpg"
    img = FileIO.load(file_path)


    println("Loaded image from: $file_path")
end


# ╔═╡ 632898bc-8d84-4d00-98f1-5934b4a6efc5
begin
    rates = [0.76666667]  # falloff strengths
    plt = plot(layout = (length(rates), 2), size=(800, 300*length(rates)))

    for (i, rate_val) in enumerate(rates)
        # call the transform with current rate
        transformed_img = expon_transform(img; rate=rate_val)

		println("rate_val = $rate_val")
		
        plot!(plt[i,1], img, title="Original", axis=nothing)
        plot!(plt[i,2], transformed_img, title="rate=$rate_val", axis=nothing)
    end

     plt
end


# ╔═╡ Cell order:
# ╠═d3ecb7f4-b248-11f0-1956-f1af898547ac
# ╠═659946da-45d6-48b5-9747-3aeaf142b28f
# ╠═61df4f3d-602d-4537-a950-399aedb86e25
# ╠═03737d26-4a70-4612-a436-69819db34416
# ╠═b87f0800-f4ea-4ece-bdea-77771a15bda7
# ╠═632898bc-8d84-4d00-98f1-5934b4a6efc5
