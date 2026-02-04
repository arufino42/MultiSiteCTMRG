
function initialise_CTMRG_open_BC(χ::Int,vA::Vector{Vector{ITensor}},r_func,List_sites::Vector{Vector{Int}})
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
            psi1=ITensor(1.,i1...)
            comb2=init_combiner(i2...,tags=tags(i2[1]))
            comb3=init_combiner(i3...,tags=tags(i3[1]))
            psi4=ITensor(1.,i4...)
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
            psi1=ITensor(1.,i1...)
            comb2=init_combiner(i2...,tags=tags(i2[1]))
            comb4=init_combiner(i4...,tags=tags(i4[1]))
            edge=*(get_A(vA,r_func,r),psi1,comb2,comb4)
            T=set_T(edge,T,r_func,r,r-u)
        end
    end
    return CTMEnvironment(χ,vA,T,C,r_func,List_sites)
end