# CLAUDE.md - AI Assistant Guide

## Project Overview

**Ronan-Project** is a research codebase exploring positional embeddings and feature mapping strategies for neural networks, with emphasis on retinal coordinate encoding and image reconstruction tasks. The project investigates how different embedding schemes (Fourier features, sparse hash-based encodings, and spatial bump maps) perform on visual signal processing and image reconstruction.

This is an academic/research project bridging computational neuroscience and deep learning.

## Repository Structure

```
Ronan-Project/
├── CLAUDE.md # This file - AI assistant guide
├── .gitignore # Excludes DATABASES directory
│
├── LogPolarRetina/ # Retinal transformation simulations
│ ├── PlotConeSpacing.jl # Cone spacing visualization
│ ├── exponRetina.jl # Exponential retinal transforms
│ ├── fisheye.jl # Fisheye lens distortion
│ ├── logpolar # Log-polar transformation
│ ├── GenerateRetinalMosaic # Retinal cone mosaic generation
│ ├── CirclesTransformed # Geometric pattern transforms
│ ├── SquaresTransformed # Square pattern transforms
│ └── logpolar.html # Visualization output
│
├── EmbeddingsForRetPos/ # Spatial hash-based embeddings
│ ├── BumpHash.jl # Spatial bump map embeddings
│ ├── BumpHash1.jl, BumpHash2.jl # Bump map variants
│ ├── Hash1vec.jl # Single vector hash encoding
│ ├── Hash1xy.jl # XY coordinate hash encoding
│ ├── ClaudeHash1.jl, ClaudeHash2.jl # Optimized hash versions
│ └── ClaudeSparse1-3.jl # Sparse hash encodings (MNIST)
│
└── TancikWithNonPeriodicEmbeddings/ # Fourier positional encoding experiments
├── Tancik1-3.jl # Core Fourier embedding implementations
├── tancik4-8*.jl # Variant implementations
├── fourier_5d_temp.jl # Fourier feature exploration
├── KevSparse3-4.jl # Sparse alternatives to Fourier
├── KevSparseBump1works.jl # Sparse + bump map hybrid
└── Tancik7.ipynb # Jupyter notebook variant

```
## Development Environment

- **Language**: Julia (v1.11+)
- **Primary Platform**: Pluto.jl interactive notebooks
- **ML Framework**: Flux.jl

### Key Dependencies

```julia
# Core ML
using Flux, MLDatasets, LinearAlgebra, Zygote

# Image Processing
using Images, ImageTransformations, ImageCore

# Visualization
using CairoMakie, Plots, Colors

# Data & Utilities
using Random, Statistics, SparseArrays, FFTW
using JLD2, Downloads, StatsBase, PlutoUI

Directory Purposes
LogPolarRetina/
Implements visual transformations simulating biological retinal properties:

Exponential scaling: Maps visual field with 268 μm per visual degree
Fisheye distortion: Variable parameter m controls strength
Log-polar transform: Foveal vision simulation
Cone spacing: Photoreceptor density using Ameln et al. (2025) formulas
EmbeddingsForRetPos/
Explores spatial hash-based embeddings for retinal coordinates:

K-Sparse Hash Encoding:

hash_k_sparse_exact(x, y, M, k, seed)  # Creates k-hot sparse vectors
color_index(x, y)                       # 4-color checkerboard position tags

Spatial Bump Maps:

SpatialBumpMapUnique(M, k, width, height, R)  # Gaussian bump grid

TancikWithNonPeriodicEmbeddings/
Primary experimental focus - Fourier positional encoding for image reconstruction:

Fourier Features:

fourier_features(x, B) = vcat(sin.(2π .* B * x), cos.(2π .* B * x))

Core Architecture:

Coordinate Input (2D)
    ↓
Feature Map (Fourier/Hash/Identity)
    ↓
Dense(hidden → 256, relu)
    ↓
Dense(256 → 256, relu)
    ↓
Dense(256 → output, sigmoid)

Code Conventions
File Naming
Tancik*: Main Fourier embedding implementations (numbered variants)
tancik*: Lowercase iteration/refinement variants
Kev*: Kevin's implementations
Claude*: Optimized or Claude-refined versions
Sparse: Sparse hash-based implementations
Hash: Hash-based spatial encoding
Bump: Gaussian bump map approaches
*works: Proven/stable implementations
Pluto Notebook Structure
# ╔═╡ [UUID]
begin
    # Cell implementation
end

Cells have unique UUIDs for reactivity
@bind macro for interactive UI elements
Embedded TOML for reproducible environments
Typical Hyperparameters
Fourier Embeddings:

mapping_size = 256      # Fourier modes
σ = 10.0f0              # Spectral scale
nepochs = 100-1000      # Training epochs
learning_rate = 1f-4    # Adam optimizer

Sparse Hash Embeddings:

M = 500-1000            # Embedding dimension
k = 6-50                # Active bits (sparsity)
seed = 0x12345678       # Deterministic hash seed

Image Processing:

crop_size = 100         # Image crop size
H, W = 100, 100         # Grid dimensions
num_scales = 2          # Multi-scale hash levels

Common Data Flow
Input: Image (URL/MNIST)
    ↓
Coordinate Grid: (2, H×W) normalized positions
    ↓
Feature Mapping: Fourier/Hash/Bump embedding
    ↓
Neural Network: Dense layers via Flux
    ↓
Output: RGB predictions (sigmoid, 0-1)
    ↓
Loss: MSE(predictions, targets)
    ↓
Training: Adam optimizer
    ↓
Evaluation: PSNR on train/test splits

Key Research Questions
How do Fourier positional encodings compare to sparse hash-based approaches?
Can sparse k-hot encodings preserve locality like dense Fourier features?
How do retinal transformations affect embedding quality?
What is the optimal embedding dimension (M) vs sparsity (k) trade-off?
Can bump maps provide biologically-inspired spatial encodings?
How well do embeddings generalize when trained on subsampled pixels?
Working with This Repository
For AI Assistants
Pluto-centric workflow: Most files are Pluto notebooks (.jl with UUID cell markers). They can be run in Pluto.jl or as regular Julia scripts.

Iterative variants: Multiple numbered variants (Tancik1-8, KevSparse3-4) represent systematic exploration. Don't consolidate without understanding the differences.

"works" suffix: Files ending in works (e.g., KevSparseBump1works.jl) are proven implementations. Treat these as stable references.

Large sparse files: ClaudeSparse* files may be very large due to extensive experiments. Handle with care.

When helping with experiments:

Preserve the Pluto cell structure (UUIDs)
Keep Float32 types for GPU compatibility
Maintain reproducibility via explicit Random.seed!() calls
Use JLD2 for model checkpointing
Running Notebooks
In Pluto:

using Pluto
Pluto.run()
# Then open the .jl file in the browser interface

As Julia Script:

include("path/to/file.jl")

Typical Modifications
Adjusting embedding dimensions and sparsity levels
Changing network architecture (layer sizes, depth)
Testing on different images or datasets
Comparing embedding strategies side-by-side
Modifying spectral scale (σ) for Fourier features
Git Workflow
DATABASES/ excluded: Large data files are gitignored
Model checkpoints: Saved via JLD2, may need local storage
Branch strategy: Feature branches for new experiments
Important Notes
This is research code, not production software
Results may vary between runs (use seeds for reproducibility)
GPU acceleration available via CUDA.jl (code uses Float32)
Visualizations use CairoMakie for publication quality
Project bridges computational neuroscience and deep learning
