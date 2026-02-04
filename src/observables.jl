function average_local_observable(net::CTMEnvironment,O::Vector{ITensor},r::Vector{Int},Lx::Int,Ly::Int)
    vT_1=[]
    vT_2=[]

    # Calculate T1
    push!(vT_1,get_C(net,1,r))
    for x=1:Lx
        push!(vT_1,get_T(net,1,r+[x,0]))
    end
    push!(vT_1,get_C(net,2,r+[Lx+1,0]))
    for y=1:Ly
        push!(vT_1,get_T(net,4,r+[0,y]))
        for x=1:Lx
            push!(vT_1,get_A(net,r+[x,y]))
        end
        push!(vT_1,get_T(net,2,r+[Lx+1,y]))
    end
    push!(vT_1,get_C(net,4,r+[0,Ly+1]))
    for x=1:Lx
        push!(vT_1,get_T(net,3,r+[x,Ly+1]))
    end
    push!(vT_1,get_C(net,3,r+[Lx+1,Ly+1]))
    
    
    # Calculate T2
    push!(vT_2,get_C(net,1,r))
    for x=1:Lx
        push!(vT_2,get_T(net,1,r+[x,0]))
    end
    push!(vT_2,get_C(net,2,r+[Lx+1,0]))
    for y=1:Ly
        push!(vT_2,get_T(net,4,r+[0,y]))
        for x=1:Lx
            push!(vT_2,translate_operator(O[net.r_func([x,y])],r+[x,y]))
        end
        push!(vT_2,get_T(net,2,r+[Lx+1,y]))
    end
    push!(vT_2,get_C(net,4,r+[0,Ly+1]))
    for x=1:Lx
        push!(vT_2,get_T(net,3,r+[x,Ly+1]))
    end
    push!(vT_2,get_C(net,3,r+[Lx+1,Ly+1]))

    
    x1=contract(vT_1)[]
    x2=contract(vT_2)[]
    return x2/x1
end


function partition_function_per_unit_cell(net::CTMEnvironment,r,Lx,Ly)
    vT_1=Union{ITensor,Vector{ITensor}}[]
    vT_2=Union{ITensor,Vector{ITensor}}[]
    vT_3=Union{ITensor,Vector{ITensor}}[]
    vT_4=Union{ITensor,Vector{ITensor}}[]
    # Calculate T2
    push!(vT_2,get_C(net,1,r))
    push!(vT_2,get_C(net,2,r+[1,0]))
    push!(vT_2,get_C(net,3,r+[1,1]))
    push!(vT_2,get_C(net,4,r+[0,1]))

    # Calculate T1
    push!(vT_1,get_C(net,1,r))
    for x=1:Lx
        push!(vT_1,get_T(net,1,r+[x,0]))
    end
    push!(vT_1,get_C(net,2,r+[Lx+1,0]))
    for y=1:Ly
        push!(vT_1,get_T(net,4,r+[0,y]))
        for x=1:Lx
            push!(vT_1,get_A(net,r+[x,y]))
        end
        push!(vT_1,get_T(net,2,r+[Lx+1,y]))
    end
    push!(vT_1,get_C(net,4,r+[0,Ly+1]))
    for x=1:Lx
        push!(vT_1,get_T(net,3,r+[x,Ly+1]))
    end
    push!(vT_1,get_C(net,3,r+[Lx+1,Ly+1]))

    # Calculate T3
    push!(vT_3,get_C(net,1,r))
    push!(vT_3,get_C(net,4,r+[0,1]))
    for x=1:Lx
        push!(vT_3,get_T(net,1,r+[x,0]))
        push!(vT_3,get_T(net,3,r+[x,1]))
    end
    push!(vT_3,get_C(net,2,r+[Lx+1,0]))
    push!(vT_3,get_C(net,3,r+[Lx+1,1]))

    # Calculate T4
    push!(vT_4,get_C(net,1,r))
    push!(vT_4,get_C(net,2,r+[1,0]))
    for y=1:Ly
        push!(vT_4,get_T(net,4,r+[0,y]))
        push!(vT_4,get_T(net,2,r+[1,y]))
    end
    push!(vT_4,get_C(net,4,r+[0,Ly+1]))
    push!(vT_4,get_C(net,3,r+[1,Ly+1]))

    x1=contract(vT_1)[]
    x2=contract(vT_2)[]
    x3=contract(vT_3)[]
    x4=contract(vT_4)[]
    return (x1*x2)/(x3*x4)
end


function calc_z(net::CTMEnvironment,r)
    Z=get_C(net,1,r+[-1,-1])*get_T(net,1,r+[0,-1])*get_C(net,2,r+[1,-1])*get_T(net,4,r+[-1,0])*get_A(net,r)*get_T(net,2,r+[1,0])*get_C(net,4,r+[-1,1])*get_T(net,3,r+[0,1])*get_C(net,3,r+[1,1])
    return Z[]
end
