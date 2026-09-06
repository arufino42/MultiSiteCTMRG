"""
    MultiSiteCTMRG

Directional corner transfer matrix renormalization group for periodic square
tensor networks with multiple inequivalent sites and one or more tensor
layers per site. See [`CTMEnvironment`](@ref) for the coordinate convention
and [`initialise_CTMRG_open_BC`](@ref) to build an environment.

Keep the state returned by [`iterate_ctmrg`](@ref) and supply a convergence
loop. Importing the module also defines multiplication between ITensors and
vectors of ITensors, and disables ITensors' tensor-order warnings globally.
"""
module MultiSiteCTMRG

export 
    CTMEnvironment,
    iterate_ctmrg,
    translate_operator,
    partition_function_per_unit_cell,
    average_local_observable,
    initialise_CTMRG_open_BC,
    get_A,
    get_T,
    get_C,
    combiner_with_memory,
    to_gpu,
    to_cpu,
    transfer_matrix_eigenvalues

using ITensors
using LinearAlgebra
using CUDA
using Adapt
using Setfield
using KrylovKit

import Base: *

"""
    *(A::ITensor, B::Vector{ITensor})

Contract `A` with the tensors in `B`, expanding the vector as `*(A, B...)`.
This method is added to `Base.*` when MultiSiteCTMRG is loaded.
"""
*(A::ITensor,B::Vector{ITensor}) = *(A,B...)
"""
    *(A::Vector{ITensor}, B::ITensor)

Contract all tensors in `A` with `B`, expanding the vector as `*(A..., B)`.
This method is added to `Base.*` when MultiSiteCTMRG is loaded.
"""
*(A::Vector{ITensor},B::ITensor) = *(A...,B)
"""
    *(A::Vector{ITensor}, B::Vector{ITensor})

Contract the tensors from both vectors as `*(A..., B...)`. This is tensor
contraction, not elementwise vector multiplication. The method is added to
`Base.*` when MultiSiteCTMRG is loaded.
"""
*(A::Vector{ITensor},B::Vector{ITensor}) = *(A...,B...)

"""
    CTMEnvironment(χ, A, T, C, r_func, List_sites)

Store a periodic multi-site CTM environment. Usually construct it with
[`initialise_CTMRG_open_BC`](@ref).

- `χ::Int`: maximum retained projector bond dimension.
- `A::Vector{Vector{ITensor}}`: local tensor layers for each inequivalent site.
- `T::Matrix{ITensor}`, `C::Matrix{ITensor}`: edges and corners, each of size
  `(4, length(List_sites))`.
- `r_func(r)`: map an integer coordinate vector `[x, y]`, including positions
  outside the reference cell, to a site index.
- `List_sites::Vector{Vector{Int}}`: one representative coordinate per site,
  ordered so that `r_func(List_sites[k]) == k`.

All stored tensors use coordinates relative to the common reference
`[1, 1]`, even for other inequivalent sites. Spatial legs have integer
coordinate tags `"x=...,y=..."`. Accessors translate these tags while
preserving index identity and prime level. Neighboring layers must share
the appropriate index IDs and translated tags.

Direction numbers `1:4` mean up, right, down, left for `T`, and upper left,
upper right, lower right, lower left for `C`. Coordinates increase rightward
and downward. The container is immutable, but holds mutable arrays. Retain
returned environments rather than assigning its fields directly.
"""
struct CTMEnvironment
    χ::Int
    A::Vector{Vector{ITensor}}
    T::Array{ITensor,2}
    C::Array{ITensor,2}
    r_func
    List_sites::Vector{Vector{Int}}
end

ITensors.disable_warn_order()

# Utility functions
include("utils.jl")
include("observables.jl")
include("initialise.jl")

# Implementation of multi-site CTMRG
include("iterate.jl")

end # module MultiSiteCTMRG