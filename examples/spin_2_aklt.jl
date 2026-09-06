using ITensors
using LinearAlgebra
using PyCall
sympy=pyimport("sympy")
@pyimport sympy.physics.quantum.cg as sympy_cg

function create_aklt_exact_peps(Lx::Int,Ly::Int)
    get_j(uj)=(uj-1)/2
    get_m(uj,um)=-get_j(uj)+um-1
    um=Index(5)
    uj=Index(5)
    # a=ITensor(prime(um,0),prime(um,1),prime(um,2),prime(um,3))
    cg_tensor=ITensor(uj,um,uj',um',uj'',um'')
    for um1 in 1:5,um2 in 1:5,um3 in 1:5,uj1 in 1:5,uj2 in 1:5,uj3 in 1:5
        j1=get_j(uj1)
        j2=get_j(uj2)
        j3=get_j(uj3)
        m1=get_m(uj1,um1)
        m2=get_m(uj2,um2)
        m3=get_m(uj3,um3)
        cg_tensor[uj1,um1,uj2,um2,uj3,um3]=convert(ComplexF64,sympy_cg.CG(j1,m1,j2,m2,j3,m3).doit().evalf())
    end
    A=cg_tensor*prime(cg_tensor,2)*prime(cg_tensor,4)
    taux_1=onehot(uj=>2)
    taux_2=onehot(uj=>5)
    a=Index(2)
    taux_3=onehot(um=>1,a=>1)+onehot(um=>2,a=>2)
    A=A*taux_1*prime(taux_1,1)*prime(taux_1,3)*prime(taux_1,5)*prime(taux_2,6)
    A=A*taux_3*prime(taux_3,1)*prime(taux_3,3)*prime(taux_3,5)
    A=mapprime(A,3=>2,5=>3,6=>0)

    r_func(r) = mod(r[1]-1,Lx)+1+Lx*mod(r[2]-1,Ly)
    N_non_eq_sites=Lx*Ly
    i=Index(2)
    j=Index(2)
    A=replaceinds(A,[a,a',a'',a''',um],[settags(j,"x=1,y=0"),settags(i,"x=1,y=1"),settags(j,"x=1,y=1"),settags(i,"x=0,y=1"),settags(um,"x=1,y=1")])

    U_ry=ITensor(
    [
        0 0 0 0 -1;
        0 0 0 1 0;
        0 0 -1 0 0;
        0 1 0 0 0;
        -1 0 0 0 0;
    ]    
    ,settags(um,"x=1,y=1"),settags(um,"x=1,y=1")')
    # @show U_ry*U_ry'
    List_sites=[[mod(k-1,Lx)+1,(k-1)÷Ly+1] for k in 1:N_non_eq_sites]
    # @show List_sites
    vA=[
        mod(sum(r),2)==0 ? copy(A) : mapprime(A*U_ry,1=>0)
    for r in List_sites]
    # @show inds(vA[1])
    # @show inds(vA[2])
    # @show inds(vA[3])
    # @show inds(vA[4])
    vλR=[diag_itensor(ones(Float64,2),settags(i,"x=1,y=1"),settags(i,"x=1,y=1")') for k in 1:N_non_eq_sites]
    vλD=[diag_itensor(ones(Float64,2),settags(j,"x=1,y=1"),settags(j,"x=1,y=1")') for k in 1:N_non_eq_sites]
    
    # vA=[copy()]
    # error("stop")
    return ipeps_su(vA,2,vλR,vλD,i,j,um,r_func,N_non_eq_sites,List_sites,[])
end

function aklt_energy(β::Float64,s::Index)
    S=create_spin_operators(2)

    x=ITensor(settags(s,"1")',settags(s,"2")',settags(s,"1"),settags(s,"2"))
    for s1p in 1:5,s2p in 1:5,s1 in 1:5, s2 in 1:5
        x[s1p,s2p,s1,s2]=sum([S[a][s1p,s1]* S[a][s2p,s2] for a in 1:3])
    end
    x2=mapprime(x*x',2=>1)
    x3=mapprime(x2*x',2=>1)
    x4=mapprime(x3*x',2=>1)
    H=(x + 7/10 * x2 + 7/45 * x3 + 1/90 * x4)/28
    U=exp(-β*H)
    comb=combiner(settags(s,"1"),settags(s,"2"))
    H_mat=array(H*comb*comb')
    x_mat=array(x*comb*comb')
    # @show H_mat
    @show eigvals(x_mat)
    @show eigvals(H_mat)
    @show size(H_mat)
    # @show H
    # @show U
    return (U,H)
end

function ctmrg_aklt_dimer_operator(r::Vector{Int},curNet::ctm_environment,peps::ipeps_su)
    dr=[1,0]
    r1=r
    r2=r1+dr
    S=create_spin_operators(2)
    x=ITensor(settags(peps.s,"x=$(r1[1]),y=$(r1[2])")',settags(peps.s,"x=$(r2[1]),y=$(r2[2])")',settags(peps.s,"x=$(r1[1]),y=$(r1[2])"),settags(peps.s,"x=$(r2[1]),y=$(r2[2])"))
    for s1p in 1:5,s2p in 1:5,s1 in 1:5, s2 in 1:5
        x[s1p,s2p,s1,s2]=sum([S[a][s1p,s1]* S[a][s2p,s2] for a in 1:3])
    end
    
    
    A1=get_A_with_weights(peps,r1)
    A2=get_A_with_weights(peps,r2)
    
    D=A1*A2*x*conj(A1')*conj(A2')
    # @show inds(D)
    unprimed_inds= filter(x->hasplev(x,0),inds(D))
    v_comb=[(
        comb=combiner(i,i',tags="");
        (
            hasid(i,id(peps.j)) ? 
            replaceind(comb,combinedind(comb),settags(curNet.j,tags(i))) :  
            replaceind(comb,combinedind(comb),settags(curNet.i,tags(i)))
        )    
    )
    for i in unprimed_inds]
    D=*(D,v_comb...)
    return D
end

function aklt_dimer_correlator(r_max::Int, curNet::ctm_environment,peps::ipeps_su)
    d_net_beg=*(
        get_C(curNet,1,[0,0]),
        get_T(curNet,4,[0,1]),
        get_C(curNet,4,[0,2]),
        get_T(curNet,1,[1,0]),
        get_T(curNet,1,[2,0]),
        ctmrg_aklt_dimer_operator([1,1],curNet,peps),
        get_T(curNet,3,[1,2]),
        get_T(curNet,3,[2,2])
    )
    z_net_beg=*(
        get_C(curNet,1,[0,0]),
        get_T(curNet,4,[0,1]),
        get_C(curNet,4,[0,2]),
        get_T(curNet,1,[1,0]),
        get_T(curNet,1,[2,0]),
        get_A(curNet,[1,1]),
        get_A(curNet,[2,1]),
        get_T(curNet,3,[1,2]),
        get_T(curNet,3,[2,2])
    )
    # @show inds(translate_operator(d_net_end,[10,1]))
    v_corr=Float64[]
    d_loc=*(
        d_net_beg,
        get_C(curNet,2,[3,0]),
        get_T(curNet,2,[3,1]),
        get_C(curNet,3,[3,2])
    )[]
    z_loc=*(
        z_net_beg,
        get_C(curNet,2,[3,0]),
        get_T(curNet,2,[3,1]),
        get_C(curNet,3,[3,2])
    )[]
    
    av=d_loc/z_loc
    @show av
    @show d_loc
    @show z_loc
    for r in 1:r_max
        c=*(
            d_net_beg,
            get_T(curNet,1,[2+r,0]),
            get_T(curNet,1,[3+r,0]),
            ctmrg_aklt_dimer_operator([2+r,1],curNet,peps),
            get_T(curNet,3,[2+r,2]),
            get_T(curNet,3,[3+r,2]),
            get_C(curNet,2,[4+r,0]),
            get_T(curNet,2,[4+r,1]),
            get_C(curNet,3,[4+r,2])
        )
        z=*(
            z_net_beg,
            get_T(curNet,1,[2+r,0]),
            get_T(curNet,1,[3+r,0]),
            get_A(curNet,[2+r,1]),
            get_A(curNet,[3+r,1]),
            get_T(curNet,3,[2+r,2]),
            get_T(curNet,3,[3+r,2]),
            get_C(curNet,2,[4+r,0]),
            get_T(curNet,2,[4+r,1]),
            get_C(curNet,3,[4+r,2])
        )
        push!(v_corr,c[]/z[])
        d_net_beg=*(
            d_net_beg,
            get_T(curNet,1,[2+r,0]),
            get_A(curNet,[2+r,1]),
            get_T(curNet,3,[2+r,2])
        )
        z_net_beg=*(
            z_net_beg,
            get_T(curNet,1,[2+r,0]),
            get_A(curNet,[2+r,1]),
            get_T(curNet,3,[2+r,2])
        )
    end
    
    return v_corr
end

begin
    τ=1e-2
    peps_aklt_exact=create_aklt_exact_peps(2,2)
    add_gates_aklt(peps_aklt_exact,τ)
    vχ=[10,20,30,40]
    net_aklt_exact=run_ctmrg(peps_aklt_exact,vχ;max_iter=3000)
    
    # convert(ComplexF64,sympy_cg.CG(1/2,1/2,1/2,-1/2,0,0).doit().evalf())
    # cb_coef=sympy.physics.quantum.cg
end

# Checking that the assymptotic dimer-dimer correlation is equal to 12.663307... (compare with Juraj's result)
begin
    
    dimer_corr=aklt_dimer_correlator(20,net_aklt_exact,peps_aklt_exact)
end