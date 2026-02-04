"""
    Auxiliary function that calculates the projector for a CTMRG move in direction `u` centered around site `r`.
"""
function get_P(net::CTMEnvironment,r::Vector{Int},u::Vector{Int};use_gpu=true)
    if use_gpu && CUDA.functional()
        v=[u[2],-u[1]]
        M1=*(
            to_gpu(get_C(net,r+u-v,r)),
            to_gpu(get_T(net,r+u,r)),
            to_gpu(get_T(net,r-v,r)),
            to_gpu.(get_A(net,r))
        )
        M2=*(
            to_gpu(get_C(net,r-2u-v,r-u)),
            to_gpu(get_T(net,r-2u,r-u)),
            to_gpu(get_T(net,r-u-v,r-u)),
            to_gpu.(get_A(net,r-u)) 
        )
        M3=*(
            to_gpu(get_C(net,r+2v+u,r+v)),
            to_gpu(get_T(net,r+v+u,r+v)),
            to_gpu(get_T(net,r+2v,r+v)),
            to_gpu.(get_A(net,r+v)) 
        )
        M4=*(
            to_gpu(get_C(net,r+2v-2u,r+v-u)),
            to_gpu(get_T(net,r+v-2u,r+v-u)),
            to_gpu(get_T(net,r+2v-u,r+v-u)),
            to_gpu.(get_A(net,r+v-u)) 
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
        @assert inds(Y)==inds(S)
        P1=R1*conj(U)*Y
        P1=replaceind(P1,commonind(P1,S),new_ind)
        P2=R2*conj(V)*Y
        P2=replaceind(P2,commonind(P2,S),new_ind)
    else
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
        X=map(x->1/sqrt(x),S*ITensor(1.,inds(S)[1]))'
        Y=X*delta(inds(S)...,inds(X)...)
        @assert inds(Y)==inds(S)
        P1=R1*conj(U)*Y
        P1=replaceind(P1,commonind(P1,S),new_ind)
        P2=R2*conj(V)*Y
        P2=replaceind(P2,commonind(P2,S),new_ind)
    end
    return [P1,P2]
end

"""
    Function which calculates an iteration of multi-site CTMRG. Choose option `gpu=true` for tensor
    contractions using GPU.
"""
function iterate_ctmrg(net::CTMEnvironment;use_gpu=true)
    if use_gpu
        for u in [[-1,0], [0,-1], [1,0], [0,1]]
            vP = [get_P(net,r,u;use_gpu=use_gpu) for r in net.List_sites]
            
            net2=net
            for r in net.List_sites
                v=[u[2],-u[1]]
                nC1=*(
                    to_gpu(get_C(net,r-v+u,r)),
                    to_gpu(get_T(net,r-v,r)),
                    translate_P(net,vP[net.r_func(r-v)][2],r-v)
                )|>to_cpu
                nT=*(
                    to_gpu(get_T(net,r+u,r)),
                    to_gpu.(get_A(net,r)),
                    vP[net.r_func(r)][2] ,
                    translate_P(net,vP[net.r_func(r-v)][1],r-v)
                )|>to_cpu
                nC2=*(
                    to_gpu(get_C(net,r+v+u,r)),
                    to_gpu(get_T(net,r+v,r)),
                    vP[net.r_func(r)][1]
                )|>to_cpu
                net2=set_C(nC1/maximum(abs.(array(nC1))),net2,r-v,r-u)
                net2=set_T(nT/maximum(abs.(array(nT))),net2,r,r-u)
                net2=set_C(nC2/maximum(abs.(array(nC2))),net2,r+v,r-u)
            end
            net=net2
        end
    else
        for u in [[-1,0], [0,-1], [1,0], [0,1]]
            vP = [get_P(net,r,u;use_gpu=use_gpu) for r in net.List_sites]
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
    end
    return net
end