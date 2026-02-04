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
    trial_func

using ITensors
using LinearAlgebra
using CUDA
using Adapt
using Setfield

import Base: *

*(A::ITensor,B::Vector{ITensor}) = *(A,B...)
*(A::Vector{ITensor},B::ITensor) = *(A...,B)
*(A::Vector{ITensor},B::Vector{ITensor}) = *(A...,B...)

function trial_func()
    println("Trial 2")
end

function sec_trial()
    println("This is a new message")
end

"""
Structure holding the Corner Transfer Matrix environment tensors.
"""
struct CTMEnvironment
    χ::Int
    A::Vector{Vector{ITensor}}
    T::Array{ITensor,2}
    C::Array{ITensor,2}
    r_func
    List_sites::Vector{Vector{Int}}
end

# Utility functions
include("utils.jl")
include("observables.jl")
include("initialise.jl")

# Implementation of multi-site CTMRG
include("iterate.jl")

end # module MultiSiteCTMRG