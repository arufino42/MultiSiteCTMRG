
"""
    initialise_CTMRG_open_BC(χ::Int, vA::Vector{Vector{ITensor}},
                             r_func, List_sites::Vector{Vector{Int}};
                             use_gpu=true) -> CTMEnvironment

Construct an open-boundary CTM seed from local tensor layers. Supply one
nonempty vector of layers per inequivalent site, stored in reference
coordinates `[1, 1]`. `r_func` must map every integer lattice coordinate
to a valid entry, with `r_func(List_sites[k]) == k`.

Each layer must share a matching virtual index with the corresponding layer
at each nearest neighbor. Layer counts and connectivity must be consistent.
All indices accessed through `get_A` need integer `x=...,y=...` tags;
on-site indices shared by layers are contracted when the layers multiply.

Outward legs are contracted with all-ones tensors, and remaining boundary
legs are fused across layers with a shared remembered combiner. Initial
corners and edges are neither normalized nor truncated to `χ`; this cutoff
is applied by subsequent CTMRG sweeps.

When `use_gpu && CUDA.functional()`, newly created boundary factors use GPU
storage. The input `vA` is retained as supplied, so move its tensors to the
GPU beforehand. Use `use_gpu=false` with CPU tensors for a CPU calculation.
This constructs a seed, not a converged environment.
"""
function initialise_CTMRG_open_BC(χ::Int,vA::Vector{Vector{ITensor}},r_func,List_sites::Vector{Vector{Int}};use_gpu=true)
    C=Array{ITensor,2}(undef,4,length(List_sites))
    T=Array{ITensor,2}(undef,4,length(List_sites))
    init_combiner=combiner_with_memory()
    # First, calculate all of the corners. Need to combine the virtual indices from all the layers. Already combined indices
    # are saved in a closure.
    for r in List_sites
        for u in [[-1,-1],[1,-1],[1,1],[-1,1]]
            v=[-u[2],u[1]]
            i1=[commonind(
                get_A(vA,r_func,r)[z],
                get_A(vA,r_func,Int.(r+(u+v)/2))[z]
            ) for z in eachindex(vA[r_func(r)])]
            i2=[commonind(
                get_A(vA,r_func,r)[z],
                get_A(vA,r_func,Int.(r+(-u+v)/2))[z]
            ) for z in eachindex(vA[r_func(r)])]
            i3=[commonind(
                get_A(vA,r_func,r)[z],
                get_A(vA,r_func,Int.(r+(-u-v)/2))[z]
            ) for z in eachindex(vA[r_func(r)])]
            i4=[commonind(
                get_A(vA,r_func,r)[z],
                get_A(vA,r_func,Int.(r+(u-v)/2))[z]
            ) for z in eachindex(vA[r_func(r)])]
            if use_gpu && CUDA.functional()
                psi1=ITensor(1.,i1...) |> to_gpu
                comb2=init_combiner(i2...,tags=tags(i2[1])) |> to_gpu
                comb3=init_combiner(i3...,tags=tags(i3[1])) |> to_gpu
                psi4=ITensor(1.,i4...) |> to_gpu
            else
                psi1=ITensor(1.,i1...)
                comb2=init_combiner(i2...,tags=tags(i2[1]))
                comb3=init_combiner(i3...,tags=tags(i3[1]))
                psi4=ITensor(1.,i4...)
            end
            corner=*(get_A(vA,r_func,r),psi1,comb2,comb3,psi4)
            C=set_C(corner,C,r_func,r,r-u)
        end 
    end

    # Next, calculate all of the edges.
    for r in List_sites
        for u in [[0,-1],[1,0],[0,1],[-1,0]]
            v=[-u[2],u[1]]
            i1=[commonind(
                get_A(vA,r_func,r)[z],
                get_A(vA,r_func,r+u)[z]
            ) for z in eachindex(vA[r_func(r)])]
            i2=[commonind(
                get_A(vA,r_func,r)[z],
                get_A(vA,r_func,r+v)[z]
            ) for z in eachindex(vA[r_func(r)])]
            i4=[commonind(
                get_A(vA,r_func,r)[z],
                get_A(vA,r_func,r-v)[z]
            ) for z in eachindex(vA[r_func(r)])]
            if use_gpu && CUDA.functional()
                psi1=ITensor(1.,i1...)  |> to_gpu
                comb2=init_combiner(i2...,tags=tags(i2[1])) |> to_gpu
                comb4=init_combiner(i4...,tags=tags(i4[1])) |> to_gpu
            else
                psi1=ITensor(1.,i1...) 
                comb2=init_combiner(i2...,tags=tags(i2[1]))
                comb4=init_combiner(i4...,tags=tags(i4[1]))
            end
            edge=*(get_A(vA,r_func,r),psi1,comb2,comb4)
            T=set_T(edge,T,r_func,r,r-u)
        end
    end
    return CTMEnvironment(χ,vA,T,C,r_func,List_sites)
end