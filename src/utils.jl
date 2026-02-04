
function sum_tag(i::Index,r::Vector{Int})
    tag0=String.(tags(i))
    regexp=r"x=(.+)"
    x=parse(Int,match(regexp,tag0[1]).captures[1])
    regexp=r"y=(.+)"
    y=parse(Int,match(regexp,tag0[2]).captures[1])
    return replacetags(i,tags(i),"x=$(x+r[1]),y=$(y+r[2])")
end


function get_A(net::CTMEnvironment,r::Vector{Int})
    v=[
    (
        new_inds=map(ind->sum_tag(ind,r.-[1,1]),inds(tensor));
        replaceinds(tensor,inds(tensor),new_inds)
    ) for tensor in net.A[net.r_func(r)]
    ]
    return v
end

function get_A(vA::Vector{Vector{ITensor}},r_func,r::Vector{Int})
    v=[
    (
        new_inds=map(ind->sum_tag(ind,r.-[1,1]),inds(tensor));
        replaceinds(tensor,inds(tensor),new_inds)
    ) for tensor in vA[r_func(r)]
    ]
    return v
end

function translate_index(i::Index,r::Vector{Int})
    return sum_tag(i,r-[1,1])
end


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
    Get the T tensor from CTMEnvironment with direction theta at position r.
"""
function get_T(net::CTMEnvironment,theta::Int,r::Vector{Int})
    k=net.r_func(r)
    new_inds=map(ind->sum_tag(ind,r.-[1,1]),inds(net.T[theta,k]))
    out=replaceinds(net.T[theta,k],inds(net.T[theta,k]),new_inds)
    return out
end


"""
    Get the T tensor from CTMEnvironment in position r around position ra.
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
    Set the T tensor in a CTMEnvironment in position r around position ra.
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
    Set the T tensor in a Matrix{ITensor} in position r around position ra.
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
    Get the C tensor with direction theta at position r.
"""
function get_C(net::CTMEnvironment,theta::Int,r::Vector{Int})
    k=net.r_func(r)
    new_inds=map(ind->sum_tag(ind,r.-[1,1]),inds(net.C[theta,k]))
    out=replaceinds(net.C[theta,k],inds(net.C[theta,k]),new_inds)
    return out
end

"""
    Get the C tensor in position r around position ra.
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
    Set the C tensor in position r around position ra.
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

@doc """
    Creates a closure `combiner(inds...;tags="")` which reuses the `combinedind` if `inds` have been combined before.
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
    Translate a projector to a site r
"""
function translate_P(net::CTMEnvironment,P::ITensor,r::Vector{Int})
    P=translate_operator(P,-net.List_sites[net.r_func(r)]+r+[1,1])
    return P
end

function to_gpu(t::ITensor)
    return adapt(CuArray,t)
end