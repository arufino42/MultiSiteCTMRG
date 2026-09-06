
"""
    sum_tag(i::Index, r::Vector{Int}) -> Index

Shift an index's coordinate tags by displacement `r = [dx, dy]`, preserving
its ID and prime level. Expects the first two tags to parse as integer
`x=...` and `y=...`; replaces the entire tag set with those two shifted
tags, discarding any others. Internal helper.
"""
function sum_tag(i::Index,r::Vector{Int})
    tag0=String.(tags(i))
    regexp=r"x=(.+)"
    x=parse(Int,match(regexp,tag0[1]).captures[1])
    regexp=r"y=(.+)"
    y=parse(Int,match(regexp,tag0[2]).captures[1])
    return replacetags(i,tags(i),"x=$(x+r[1]),y=$(y+r[2])")
end


"""
    get_A(net::CTMEnvironment, r::Vector{Int}) -> Vector{ITensor}

Return the local layers for site `net.r_func(r)`, with all index coordinates
shifted by `r - [1, 1]`. Stored tensors are referenced to `[1, 1]`.
The returned vector contains one ITensor per layer, including in the
single-layer case. Does not mutate the stored tensors.
"""
function get_A(net::CTMEnvironment,r::Vector{Int})
    v=[
    (
        new_inds=map(ind->sum_tag(ind,r.-[1,1]),inds(tensor));
        replaceinds(tensor,inds(tensor),new_inds)
    ) for tensor in net.A[net.r_func(r)]
    ]
    return v
end

"""
    get_A(vA::Vector{Vector{ITensor}}, r_func, r::Vector{Int})

Return the layers `vA[r_func(r)]` translated from reference `[1, 1]` to
`r`, without an environment container. Every index must carry integer
coordinate tags. Used by [`initialise_CTMRG_open_BC`](@ref).
"""
function get_A(vA::Vector{Vector{ITensor}},r_func,r::Vector{Int})
    v=[
    (
        new_inds=map(ind->sum_tag(ind,r.-[1,1]),inds(tensor));
        replaceinds(tensor,inds(tensor),new_inds)
    ) for tensor in vA[r_func(r)]
    ]
    return v
end

"""
    translate_index(i::Index, r::Vector{Int}) -> Index

Shift coordinate tags by `r - [1, 1]`, retaining the index ID and prime
level. Thus `r = [1, 1]` leaves coordinates unchanged. Uses
[`sum_tag`](@ref), which replaces the entire tag set. Internal helper.
"""
function translate_index(i::Index,r::Vector{Int})
    return sum_tag(i,r-[1,1])
end


"""
    translate_operator(OP::ITensor, r::Vector{Int}) -> ITensor

Translate coordinate-tagged indices of `OP` by `r - [1, 1]`.
The argument is a destination for a tensor stored at reference `[1, 1]`,
not a displacement: use `r = displacement + [1, 1]` for an arbitrary shift.

Indices matching the `x=...,y=...` tag pattern are translated; other indices
are left unchanged. Coordinate tags must parse as integers. Translation
preserves index IDs and prime levels but replaces the full tag set on matched
indices, so additional tags are discarded. Does not mutate `OP`.

To store a tensor currently located at `position` in reference coordinates,
use `translate_operator(OP, [2, 2] - position)`.
"""
function translate_operator(OP::ITensor,r::Vector{Int})
    stags(y) = *((String.(tags(y)).|>x->x*",")...)[1:end-1]
    inds_to_change=filter(
        x->(
            m=match(r"x=(.+),y=(.+)",stags(x));
            isnothing(m) ? false : length(m.captures)==2
        )
    ,inds(OP))
    new_inds=map(ind->sum_tag(ind,r.-[1,1]),inds_to_change)
    out=replaceinds(OP,inds_to_change,new_inds)
    return out
end


"""
    get_T(net::CTMEnvironment, theta::Int, r::Vector{Int}) -> ITensor

Return edge `net.T[theta, net.r_func(r)]` translated to its position `r`.
Directions `theta = 1, 2, 3, 4` mean up, right, down, left. The coordinate
is the edge's position, not the central site's position.
"""
function get_T(net::CTMEnvironment,theta::Int,r::Vector{Int})
    k=net.r_func(r)
    new_inds=map(ind->sum_tag(ind,r.-[1,1]),inds(net.T[theta,k]))
    out=replaceinds(net.T[theta,k],inds(net.T[theta,k]),new_inds)
    return out
end


"""
    get_T(net::CTMEnvironment, r::Vector{Int}, ra::Vector{Int}) -> ITensor

Return the edge at `r` surrounding the central site `ra`. The displacement
`r - ra` must be `[0,-1]`, `[1,0]`, `[0,1]`, or `[-1,0]`;
otherwise raises an error. Selects the stored site by `net.r_func(r)`.
"""
function get_T(net::CTMEnvironment,r::Vector{Int},ra::Vector{Int})
    dr=r-ra
    if dr==[0,-1]
        theta=1
    elseif dr==[1,0]
        theta=2
    elseif dr==[0,1]
        theta=3
    elseif dr==[-1,0]
        theta=4
    else
        error("Invalid call to get_T.")
    end
    k=net.r_func(r)
    new_inds=map(ind->sum_tag(ind,r.-[1,1]),inds(net.T[theta,k]))
    out=replaceinds(net.T[theta,k],inds(net.T[theta,k]),new_inds)
    return out
end

"""
    set_T(T::ITensor, net::CTMEnvironment, r::Vector{Int}, ra::Vector{Int})

Return an updated environment with edge `T` at position `r` around `ra`.
Infer the direction from the cardinal displacement `r - ra`, select the
site with `net.r_func(r)`, and translate `T` back to reference coordinates.
Keep the returned environment. Invalid displacements raise an error.
Internal helper.
"""
function set_T(T::ITensor,net::CTMEnvironment,r::Vector{Int},ra::Vector{Int})
    dr=r-ra
    if dr==[0,-1]
        theta=1
    elseif dr==[1,0]
        theta=2
    elseif dr==[0,1]
        theta=3
    elseif dr==[-1,0]
        theta=4
    else
        error("Invalid call to get_T.")
    end
    k=net.r_func(r)
    net = @set net.T[theta,k]=translate_operator(T,-r+[2,2])
    return net
end

"""
    set_T(T::ITensor, vT::Matrix{ITensor}, r_func, r::Vector{Int}, ra::Vector{Int})

Return an updated edge matrix, storing `T` in reference coordinates in the
direction selected by `r - ra` and column `r_func(r)`. Requires a cardinal
unit displacement. Keep the returned matrix. Internal initialization helper.
"""
function set_T(T::ITensor,vT::Matrix{ITensor},r_func,r::Vector{Int},ra::Vector{Int})
    dr=r-ra
    if dr==[0,-1]
        theta=1
    elseif dr==[1,0]
        theta=2
    elseif dr==[0,1]
        theta=3
    elseif dr==[-1,0]
        theta=4
    else
        error("Invalid call to get_T.")
    end
    k=r_func(r)
    vT = @set vT[theta,k]=translate_operator(T,-r+[2,2])
    return vT
end

"""
    get_C(net::CTMEnvironment, theta::Int, r::Vector{Int}) -> ITensor

Return corner `net.C[theta, net.r_func(r)]` translated to position `r`.
Directions `1:4` mean upper left, upper right, lower right, lower left.
The coordinate is the corner's position, not the central site's position.
"""
function get_C(net::CTMEnvironment,theta::Int,r::Vector{Int})
    k=net.r_func(r)
    new_inds=map(ind->sum_tag(ind,r.-[1,1]),inds(net.C[theta,k]))
    out=replaceinds(net.C[theta,k],inds(net.C[theta,k]),new_inds)
    return out
end

"""
    get_C(net::CTMEnvironment, r::Vector{Int}, ra::Vector{Int}) -> ITensor

Return the corner at `r` surrounding central site `ra`. The displacement
`r - ra` must be `[-1,-1]`, `[1,-1]`, `[1,1]`, or `[-1,1]`;
otherwise raises an error. Selects the stored site by `net.r_func(r)`.
"""
function get_C(net::CTMEnvironment,r::Vector{Int},ra::Vector{Int})
    dr=r-ra
    if dr==[-1,-1]
        theta=1
    elseif dr==[1,-1]
        theta=2
    elseif dr==[1,1]
        theta=3
    elseif dr==[-1,1]
        theta=4
    else
        error("Invalid call to get_C.")
    end
    k=net.r_func(r)
    new_inds=map(ind->sum_tag(ind,r.-[1,1]),inds(net.C[theta,k]))
    out=replaceinds(net.C[theta,k],inds(net.C[theta,k]),new_inds)
    return out
end

"""
    set_C(C::ITensor, net::CTMEnvironment, r::Vector{Int}, ra::Vector{Int})

Return an updated environment with corner `C` at `r` around `ra`.
Infer the direction from the diagonal displacement `r - ra`, select the
site with `net.r_func(r)`, and translate `C` back to reference coordinates.
Keep the returned environment. Invalid displacements raise an error.
Internal helper.
"""
function set_C(C::ITensor,net::CTMEnvironment,r::Vector{Int},ra::Vector{Int})
    dr=r-ra
    if dr==[-1,-1]
        theta=1
    elseif dr==[1,-1]
        theta=2
    elseif dr==[1,1]
        theta=3
    elseif dr==[-1,1]
        theta=4
    else
        error("Invalid call to set_C.")
    end
    k=net.r_func(r)
    net = @set net.C[theta,k]=translate_operator(C,-r+[2,2])
    return net
end

"""
    set_C(C::ITensor, vC::Matrix{ITensor}, r_func, r::Vector{Int}, ra::Vector{Int})

Return an updated corner matrix with `C` stored in reference coordinates,
in the direction selected by `r - ra` and column `r_func(r)`. Requires
a diagonal displacement with components `±1`. Keep the returned matrix.
Internal initialization helper.
"""
function set_C(C::ITensor,vC::Matrix{ITensor},r_func,r::Vector{Int},ra::Vector{Int})
    dr=r-ra
    if dr==[-1,-1]
        theta=1
    elseif dr==[1,-1]
        theta=2
    elseif dr==[1,1]
        theta=3
    elseif dr==[-1,1]
        theta=4
    else
        error("Invalid call to set_C.")
    end
    k=r_func(r)
    vC = @set vC[theta,k]=translate_operator(C,-r+[2,2])
    return vC
end

"""
    combiner_with_memory() -> closure

Create a callable `combine(inds...; tags="")` that returns an ITensor combiner
while reusing a combined-index ID for previously encountered input indices.
Each closure owns an independent cache.

The cache key is a set of input indices with tags removed. It ignores input
order and tags, but retains IDs and prime levels. The returned combined index
gets the requested tags. Reuse the closure across translated copies of the
same bonds, and keep a consistent input ordering so that the fused basis has
a consistent interpretation.
"""
function combiner_with_memory()
    ind_list=Dict{Set{Index},Index}()
    return function (inds...;tags="")
        s=Set([settags(i,"") for i in inds])
        if !haskey(ind_list,s)
            comb=combiner(inds...;tags=tags)
            ind_list[s]=settags(combinedind(comb),"")
            return comb
        else
            comb=combiner(inds...)
            return replaceind(comb,combinedind(comb),settags(ind_list[s],tags))
        end
    end
end

"""
    translate_P(net::CTMEnvironment, P::ITensor, r::Vector{Int}) -> ITensor

Translate a projector computed at representative site
`net.List_sites[net.r_func(r)]` to the equivalent site `r`, shifting its
coordinate tags by their displacement. Internal directional-update helper.
"""
function translate_P(net::CTMEnvironment,P::ITensor,r::Vector{Int})
    P=translate_operator(P,-net.List_sites[net.r_func(r)]+r+[1,1])
    return P
end

"""
    to_gpu(t::ITensor) -> ITensor

Adapt tensor storage to CUDA `CuArray` using Adapt.jl. Requires a functional
CUDA device; this helper has no CPU fallback. Does not convert an entire
environment. See [`to_cpu`](@ref).
"""
function to_gpu(t::ITensor)
    return adapt(CuArray,t)
end

"""
    to_cpu(t::ITensor) -> ITensor

Adapt tensor storage to a CPU `Array` using Adapt.jl. Converts one tensor,
not an entire environment. See [`to_gpu`](@ref).
"""
function to_cpu(t::ITensor)
    return adapt(Array,t)
end