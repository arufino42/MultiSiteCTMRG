# CHECKED FOR BUGS
function get_R1_R2(r,theta,net::CTMEnvironment)
    if theta==4
        M1=get_C(net,1,r+[-1,-1])*get_T(net,4,r+[-1,0])*get_T(net,1,r+[0,-1])*get_A(net,r+[0,0])
        M2=get_T(net,1,r+[1,-1])*get_A(net,r+[1,0])*get_C(net,2,r+[2,-1])*get_T(net,2,r+[2,0])
        R1=M1*M2
        M3=get_C(net,4,r+[-1,2])*get_T(net,4,r+[-1,1])*get_T(net,3,r+[0,2])*get_A(net,r+[0,1])
        M4=get_T(net,3,r+[1,2])*get_A(net,r+[1,1])*get_C(net,3,r+[2,2])*get_T(net,2,r+[2,1])
        R2=M3*M4
    end
    if theta==1
        M1=get_C(net,2,r+[1,-1])*get_T(net,1,r+[0,-1])*get_T(net,2,r+[1,0])*get_A(net,r+[0,0])
        M2=get_T(net,2,r+[1,1])*get_A(net,r+[0,1])*get_C(net,3,r+[1,2])*get_T(net,3,r+[0,2])
        R1=M1*M2
        M3=get_C(net,1,r+[-2,-1])*get_T(net,1,r+[-1,-1])*get_T(net,4,r+[-2,0])*get_A(net,r+[-1,0])
        M4=get_T(net,4,r+[-2,1])*get_A(net,r+[-1,1])*get_C(net,4,r+[-2,2])*get_T(net,3,r+[-1,2])
        R2=M3*M4
    end
    return (R1,R2)
end
function get_R1_R2_OLD(r,theta,net::CTMEnvironment)
    if theta==4
        R1=get_C(net,1,r+[-1,-1])*get_T(net,4,r+[-1,0])*get_T(net,1,r+[0,-1])*get_A(net,r+[0,0])*get_T(net,1,r+[1,-1])*get_A(net,r+[1,0])*get_C(net,2,r+[2,-1])*get_T(net,2,r+[2,0])
        R2=get_C(net,4,r+[-1,2])*get_T(net,4,r+[-1,1])*get_T(net,3,r+[0,2])*get_A(net,r+[0,1])*get_T(net,3,r+[1,2])*get_A(net,r+[1,1])*get_C(net,3,r+[2,2])*get_T(net,2,r+[2,1])
    end
    if theta==1
        R1=get_C(net,2,r+[1,-1])*get_T(net,1,r+[0,-1])*get_T(net,2,r+[1,0])*get_A(net,r+[0,0])*get_T(net,2,r+[1,1])*get_A(net,r+[0,1])*get_C(net,3,r+[1,2])*get_T(net,3,r+[0,2])
        R2=get_C(net,1,r+[-2,-1])*get_T(net,1,r+[-1,-1])*get_T(net,4,r+[-2,0])*get_A(net,r+[-1,0])*get_T(net,4,r+[-2,1])*get_A(net,r+[-1,1])*get_C(net,4,r+[-2,2])*get_T(net,3,r+[-1,2])
    end
    return (R1,R2)
end

# CHECKED FOR BUGS
function get_P1_P2(r,theta,net::CTMEnvironment)
    if theta==4
        (R1,R2)=get_R1_R2(r,theta,net)
        prime!(R2,tags="x=$(r[1]+1)")
        prime!(R2,tags="x=$(r[1]+2)")
        M=R1*R2
        bd=size(M,3)*size(M,4)
        # @show size(M)
        U,S,V=svd(M,filter(x->hasplev(x,0),inds(M));maxdim=min(net.chi,bd),cutoff=1e-25)
        # @show S
        P1=R1*conj.(U)*(S.^(-1/2))
        P2=R2*V*(S.^(-1/2))
        return P1,P2
    end
    if theta==1
        (R1,R2)=get_R1_R2(r,theta,net)
        prime!(R2,tags="y=$(r[2]+1)")
        prime!(R2,tags="y=$(r[2]+2)")
        M=R1*R2
        bd=size(M,3)*size(M,4)
        U,S,V=svd(M,filter(x->hasplev(x,0),inds(M));maxdim=min(net.chi,bd),cutoff=1e-25)
        # @show S
        P1=R1*conj.(U)*(S.^(-1/2))
        P2=R2*V*(S.^(-1/2))
        return P1,P2
    end
    if theta==2
        (R2,R1)=get_R1_R2(r-[1,1],4,net)
        prime!(R2,tags="x=$(r[1]-1)")
        prime!(R2,tags="x=$(r[1]-2)")
        M=R1*R2
        bd=size(M,3)*size(M,4)
        U,S,V=svd(M,filter(x->hasplev(x,0),inds(M));maxdim=min(net.chi,bd),cutoff=1e-25)
        # @show S
        P1=R1*conj.(U)*(S.^(-1/2))
        P2=R2*V*(S.^(-1/2))
        return P1,P2
    end
    if theta==3
        (R2,R1)=get_R1_R2(r+[1,-1],1,net)
        prime!(R2,tags="y=$(r[2]-1)")
        prime!(R2,tags="y=$(r[2]-2)")
        M=R1*R2
        bd=size(M,3)*size(M,4)
        U,S,V=svd(M,filter(x->hasplev(x,0),inds(M));maxdim=min(net.chi,bd),cutoff=1e-25)
        # @show S
        P1=R1*conj.(U)*(S.^(-1/2))
        P2=R2*V*(S.^(-1/2))
        return P1,P2
    end
end


function trans_P(net::CTMEnvironment,k::Int,P::ITensor,r::Vector{Int})
    pos_inds=filter(ind->!hastags(ind,"Link"),inds(P))
    new_inds=map( ind->sum_tag(ind,r-net.List_sites[k]),pos_inds)
    out=replaceinds(P,pos_inds,new_inds)
    return out
end

function m_plus(x::Int)
    return mod(x,4)+1
end

function m_minus(x::Int)
    return mod(x-2,4)+1
end

function delta_r(k::Int)
    v=[
        [1,0],
        [0,1],
        [-1,0],
        [0,-1]
    ]
    return v[k]
end

function Delta(k::Int)
    v=[
        [0,-1],
        [1,0],
        [0,1],
        [-1,0]
    ]
    return v[k]
end

function new_environment(net::CTMEnvironment,theta)
    chi=net.chi
    A=net.A
    T=Array{ITensor,2}(undef,4,net.N_non_eq_sites)
    C=Array{ITensor,2}(undef,4,net.N_non_eq_sites)
    for k=1:net.N_non_eq_sites
        for alpha=1:4
            if alpha==theta
                T[alpha,k]=ITensor(undef)
            else
                T[alpha,k]=net.T[alpha,k]
            end
            if alpha==m_plus(theta)||alpha==theta
                C[alpha,k]=ITensor(undef)
            else
                C[alpha,k]=net.C[alpha,k]
            end
        end
    end
    i=net.i
    j=net.j
    if theta==1
        a=[Index(0) for k=1:net.N_non_eq_sites]
        b=net.b
        c=net.c
        d=net.d
    elseif theta==2
        a=net.a
        b=[Index(0) for k=1:net.N_non_eq_sites]
        c=net.c
        d=net.d
    elseif theta==3
        a=net.a
        b=net.b
        c=[Index(0) for k=1:net.N_non_eq_sites]
        d=net.d
    elseif theta==4
        a=net.a
        b=net.b
        c=net.c
        d=[Index(0) for k=1:net.N_non_eq_sites]
    end
    r_func=net.r_func
    N_non_eq_sites=net.N_non_eq_sites
    List_sites=net.List_sites
    return CTMEnvironment(chi,A,T,C,i,j,a,b,c,d,r_func,N_non_eq_sites,List_sites)
end


"""
    iterate_ctmrg(net::CTMEnvironment,verbose=false)

Calculate one iteraction of Multi-Site CTMRG. 
"""
function iterate_ctmrg(net::CTMEnvironment;verbose=false)
    if typeof(net.A[1])==Vector{ITensor}
        return new_iterate_ctmrg(net)
    end
    for theta in [4,1,2,3]
        vP1=Array{ITensor,1}(undef,net.N_non_eq_sites)
        vP2=Array{ITensor,1}(undef,net.N_non_eq_sites)
        new_inds=Vector{Index}(undef,net.N_non_eq_sites)
        for r in net.List_sites
            P1,P2=get_P1_P2(r,theta,net)
            vP1[net.r_func(r)]=P1
            vP2[net.r_func(r)]=P2
            new_inds[net.r_func(r)]=Index(dim(filter(ind->hastags(ind,"Link"),inds(vP1[net.r_func(r)]))))
        end
        # @show inds(vP1[1]*vP2[1]) 
        # @show norm(array(vP1[1]*vP2[1])-diagm(repeat([1.],dim(filter(ind->hastags(ind,"Link"),inds(vP1[1]))))))
        net2=new_environment(net,theta)
        v_inds=[:a,:b,:c,:d]
        setproperty!(net2,v_inds[theta],new_inds)
        vC_plus_tags=["x=0,y=1", "x=1,y=0", "x=1,y=1","x=1,y=1" ]
        vT_tags_v=["x=1,y=1","x=1,y=1","x=0,y=1","x=1,y=0"]
        vT_tags_u=["x=0,y=1", "x=1,y=0", "x=1,y=1","x=1,y=1" ]
        vC_minus_tags=["x=1,y=1","x=1,y=1","x=0,y=1","x=1,y=0"]
        if verbose
            println("\n\n\n CTMRG iteration")
            println("\n\n Theta=$theta")
        end
        for (k,r) in enumerate(net.List_sites)
            if verbose
                println("\n\n r=$r")
                println("First old corner:   C[$(m_plus(theta)),$(net.r_func(r+delta_r(theta)+Delta(theta)))]")
                println("First enlarged corner:   C[$(m_plus(theta)),$(net.r_func(r+delta_r(theta)))]")
                @show inds(trans_P(net,net.r_func(r+delta_r(theta)),vP2[net.r_func(r+delta_r(theta))],r+delta_r(theta)))
            end
            net2.C[m_plus(theta),net.r_func(r+delta_r(theta))]=
                get_C(net,m_plus(theta),r+delta_r(theta)+Delta(theta))*
                get_T(net,m_plus(theta),r+delta_r(theta))*
                trans_P(net,net.r_func(r+delta_r(theta)),vP2[net.r_func(r+delta_r(theta))],r+delta_r(theta))
            link_ind=filter(ind->hastags(ind,"Link"),inds(net2.C[m_plus(theta),net.r_func(r+delta_r(theta))]))[1]
            no_link_ind=filter(ind->!hastags(ind,"Link"),inds(net2.C[m_plus(theta),net.r_func(r+delta_r(theta))]))[1]
            no_link_ind2=sum_tag(no_link_ind,-(r+delta_r(theta))+[1,1])
            replaceind!(net2.C[m_plus(theta),net.r_func(r+delta_r(theta))],no_link_ind,no_link_ind2)
            replaceind!(net2.C[m_plus(theta),net.r_func(r+delta_r(theta))],link_ind,settags(getproperty(net2,v_inds[theta])[net.r_func(r+delta_r(theta))],vC_plus_tags[theta]))
            # @show maximum(abs.(array(net2.C[m_plus(theta),net.r_func(r+delta_r(theta))])))
            net2.C[m_plus(theta),net.r_func(r+delta_r(theta))]=net2.C[m_plus(theta),net.r_func(r+delta_r(theta))]/maximum(abs.(array(net2.C[m_plus(theta),net.r_func(r+delta_r(theta))])))
            
            # Update T[theta,r]
            if verbose  
                println("\n\n\n")  
                println("Enlarged edge:   T[$(theta),$k]")
                @show r+delta_r(theta)
                @show inds(vP1[net.r_func(r+delta_r(theta))])
                @show inds(trans_P(net,net.r_func(r+delta_r(theta)),vP1[net.r_func(r+delta_r(theta))],r+delta_r(theta)))
                @show inds(get_T(net,theta,r+Delta(theta)))
                @show inds(get_A(net,r))
                @show inds(trans_P(net,k,vP2[k],r))
                println("\n\n\n")  
            end
            net2.T[theta,k]=
                trans_P(net,net.r_func(r+delta_r(theta)),vP1[net.r_func(r+delta_r(theta))],r+delta_r(theta))*
                get_T(net,theta,r+Delta(theta))*
                get_A(net,r)*
                trans_P(net,k,vP2[k],r)
            no_link_ind=filter(ind->!hastags(ind,"Link"),inds(net2.T[theta,k]))[1]
            no_link_ind2=sum_tag(no_link_ind,-r+[1,1])
            v_ind=filter(ind->hastags(ind,"v"),inds(net2.T[theta,k]))[1]
            u_ind=filter(ind->hastags(ind,"u"),inds(net2.T[theta,k]))[1]
            replaceind!(net2.T[theta,k],no_link_ind,no_link_ind2)
            replaceind!(net2.T[theta,k],v_ind,settags(getproperty(net2,v_inds[theta])[net.r_func(r+delta_r(theta))],vT_tags_v[theta]))
            replaceind!(
                net2.T[theta,k],
                u_ind,
                settags(
                    getproperty(net2,v_inds[theta])[net.r_func(r)],vT_tags_u[theta]
                )
            )
            
            net2.T[theta,k]=net2.T[theta,k]/maximum(abs.(array(net2.T[theta,k])))
            
            if verbose
                println("Second enlarged corner:   C[$theta,$(net.r_func(r-delta_r(theta)))]")
            end
            # Update C[theta,r-delta_r(theta)]
            net2.C[theta,net.r_func(r-delta_r(theta))]=
                get_C(net,theta,r-delta_r(theta)+Delta(theta))*
                get_T(net,m_minus(theta),r-delta_r(theta))*
                trans_P(net,k,vP1[k],r)
            
            link_ind=filter(ind->hastags(ind,"Link"),inds(net2.C[theta,net.r_func(r-delta_r(theta))]))[1]
            no_link_ind=filter(ind->!hastags(ind,"Link"),inds(net2.C[theta,net.r_func(r-delta_r(theta))]))[1]
            no_link_ind2=sum_tag(no_link_ind,-(r-delta_r(theta))+[1,1])
            replaceind!(net2.C[theta,net.r_func(r-delta_r(theta))],link_ind,settags(getproperty(net2,v_inds[theta])[net.r_func(r)],vC_minus_tags[theta]))
            replaceind!(net2.C[theta,net.r_func(r-delta_r(theta))],no_link_ind,no_link_ind2)
            net2.C[theta,net.r_func(r-delta_r(theta))]=net2.C[theta,net.r_func(r-delta_r(theta))]/maximum(abs.(array(net2.C[theta,net.r_func(r-delta_r(theta))])))
        end
        net=net2
        # @show calc_z(net,[1,1])
        # @show calc_z(net,[1,2])
        # @show net.C[1,1]
        # @show net.C[1,2]
        # println("(a1,b,c,d)  =  ",net.a,net.b,net.c,net.d)
    end
    
   
    # println("(a1,b1,c1,d1)  =  ",net.a[1],net.b[1],net.c[1],net.d[1])
    # println("(a2,b2,c2,d2)  =  ",net.a[2],net.b[2],net.c[2],net.d[2])
    return net
end