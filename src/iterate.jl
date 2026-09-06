"""
    get_P(net::CTMEnvironment, r::Vector{Int}, u::Vector{Int}) -> Vector{ITensor}

Build the pair of projectors for a directional CTMRG move centered at `r`
along the cardinal unit vector `u`. Contracts four enlarged corners into
two half environments, then uses an SVD with `maxdim=net.χ` and
`cutoff=1e-15`. Inverse square roots of zero singular values are set to zero.

Returns `[P1, P2]` with coordinate-tagged retained indices. Does not update
the environment. Internal helper for [`iterate_ctmrg`](@ref).
"""
function get_P(net::CTMEnvironment,r::Vector{Int},u::Vector{Int})
    
    v=[u[2],-u[1]]
    M1=*(
        get_C(net,r+u-v,r),
        get_T(net,r+u,r),
        get_T(net,r-v,r),
        get_A(net,r)
    )
    M2=*(
        get_C(net,r-2u-v,r-u),
        get_T(net,r-2u,r-u),
        get_T(net,r-u-v,r-u),
        get_A(net,r-u)
    )
    M3=*(
        get_C(net,r+2v+u,r+v),
        get_T(net,r+v+u,r+v),
        get_T(net,r+2v,r+v),
        get_A(net,r+v)
    )
    M4=*(
        get_C(net,r+2v-2u,r+v-u),
        get_T(net,r+v-2u,r+v-u),
        get_T(net,r+2v-u,r+v-u),
        get_A(net,r+v-u)
    )
    R1=M1*M2
    R2=replaceinds(M3*M4,uniqueinds(M4,M3),addtags(uniqueinds(M4,M3),"*"))
    Z=R1*R2
    U,S,V=svd(Z,uniqueinds(M2,M1);
        cutoff=1e-15,
        maxdim=net.χ,
        lefttags=tags(commonind(get_A(net,r)[1],get_A(net,r+v)[1])),
        righttags=tags(commonind(get_A(net,r)[1],get_A(net,r+v)[1]))
    )
    new_ind=commonind(U,S)
    Y=map(x-> x==0. ? 0. : 1/sqrt(x),S)
    P1=R1*conj(U)*Y
    P1=replaceind(P1,commonind(P1,S),new_ind)
    P2=R2*conj(V)*Y
    P2=replaceind(P2,commonind(P2,S),new_ind)
    return [P1,P2]
end

"""
    iterate_ctmrg(net::CTMEnvironment) -> CTMEnvironment

Perform one complete directional sweep: left, up, right, down. For each
direction, compute projectors for all representative sites from the current
environment, then update their corners and edges. Retain the returned state:
`net = iterate_ctmrg(net)`.

Projector SVDs use maximum dimension `net.χ` and cutoff `1e-15`. Each
updated boundary tensor is divided by its maximum absolute entry; zero
tensors are not guarded against. Bulk tensors and the site map are retained.

No convergence test or diagnostic tuple is returned. Device placement follows
the tensor storage; there is no `gpu` or `use_gpu` keyword on this function.
Observables contract the current tensors directly, so no environment-cache
refresh is required.
"""
function iterate_ctmrg(net::CTMEnvironment)
    for u in [[-1,0], [0,-1], [1,0], [0,1]]
        vP = [get_P(net,r,u) for r in net.List_sites]
        net2=net
        for r in net.List_sites
            v=[u[2],-u[1]]
            nC1=*(
                (get_C(net,r-v+u,r)),
                (get_T(net,r-v,r)),
                translate_P(net,vP[net.r_func(r-v)][2],r-v)
            )
            nT=*(
                (get_T(net,r+u,r)),
                (get_A(net,r)),
                vP[net.r_func(r)][2] ,
                translate_P(net,vP[net.r_func(r-v)][1],r-v)
            )
            nC2=*(
                (get_C(net,r+v+u,r)),
                (get_T(net,r+v,r)),
                vP[net.r_func(r)][1]
            )
            net2=set_C(nC1/maximum(abs.(array(nC1))),net2,r-v,r-u)
            net2=set_T(nT/maximum(abs.(array(nT))),net2,r,r-u)
            net2=set_C(nC2/maximum(abs.(array(nC2))),net2,r+v,r-u)
        end
        net=net2
    end
    return net
end