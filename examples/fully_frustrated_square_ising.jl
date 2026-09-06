using MultiSiteCTMRG
using DelimitedFiles
using JLD2
using ITensors
using LinearAlgebra
using Setfield

function pos_to_site_Ising(r)
    return r[1]>0 ? (r[1]-1)%2+1 : (-r[1]+1)%2+1
end

function ff_square_energy_a(s,J::Float64)
    return J*(-s[1]*s[2]+s[2]*s[3]+s[3]*s[4]+s[4]*s[1])/2
end


function ff_square_energy_b(s,J::Float64)
    return J*(s[1]*s[2]+s[2]*s[3]-s[3]*s[4]+s[4]*s[1])/2
end


function sigma_to_s(sigma)
    L=length(sigma)
    x=0
    for k=1:L
        x+=Int(sigma[k]+1)÷2*(2^(k-1))
    end
    return x+1
end

function s_to_sigma(s,L)
    return digits(s-1,base=2,pad=L).*2 .-1
end


function ff_square_ising(J::Float64,T::Float64)
    i=Index(4)
    j=Index(4)
    indset=[
        settags(j,"x=1,y=0"),
        settags(i,"x=1,y=1"),
        settags(j,"x=1,y=1"),
        settags(i,"x=0,y=1")
    ]
    A=ITensor(indset...)
    B=ITensor(indset...)
    Egs=-J
    for s=0:2^4-1
        sigma=digits(s,base=2,pad=4).*2 .-1
        u=digits(s,base=2,pad=4) .+1
        Ea=ff_square_energy_a(sigma,J)
        Eb=ff_square_energy_b(sigma,J)
        i1=sigma_to_s([sigma[4],sigma[1]])
        i2=sigma_to_s([sigma[1],sigma[2]])
        i3=sigma_to_s([sigma[3],sigma[2]])
        i4=sigma_to_s([sigma[4],sigma[3]])
        if T==0 && abs(Ea-Egs)<=1e-7
            A[i1,i2,i3,i4]+=1
        elseif T>0
            A[i1,i2,i3,i4]+= exp(-(Ea-Egs)/T)
        end
        if T==0 && abs(Eb-Egs)<=1e-7
            B[i1,i2,i3,i4]+=1
        elseif T>0
            B[i1,i2,i3,i4]+= exp(-(Eb-Egs)/T)
        end
    end
    return [[A],[B]]
end


# Models: SquareIsing, FFIsing
Tc_ising=1/(log(1+sqrt(2))/2)
J=1.
T=0.
chi=10
boundaryCondition=:Open
model=:FFIsing
ftol=1e-11
deltaF=2*ftol
convF=[]
Nmax=10000
Nmin=100
i=0
vf=[] 
ve=[]
vm=[]
curNet=nothing
folder=mkpath("examples/results/fully_frustrated_ising")*"/"

vA=ff_square_ising(J,T)
site_func(r)=mod(r[1],2)+1
List_sites=[[2,1],[1,1]]

for chi in 10:10:50
    curNet=initialise_CTMRG_open_BC(chi,vA,site_func,List_sites)
    curNet = @set curNet.χ=chi
    println("Changing to chi = $chi")
    deltaF=2*ftol
    i=0
    k1=0.
    k2=0.
    f2=0
    f1=0
    e2=0
    m2=0
    Nmin=1
    Nmax=10000
    deltaF=2*ftol
    while (abs(deltaF)>=ftol && i<Nmax) || i<Nmin
        @show T
        i+=1
        curNet=iterate_ctmrg(curNet);
        
        k2=partition_function_per_unit_cell(curNet,[0,0],2,2)
        deltaK=k2-k1
        k1=k2
        try
            f2=log(k2)/4
        catch 
            f2=0
        end
        deltaF=f2-f1
        f1=f2
        # e2=average_local_observable(curNet,E_op,[0,0],1,1)
        # m2=average_local_observable(curNet,M_op,[0,0],1,1)
        println("f = ",f2)
        println("m = ",m2)
        println("deltaF = ",deltaF)
        push!(convF,(f2))
    end
    dataPoint=Dict()
    dataPoint["f"]=f2
    dataPoint["m"]=m2
    dataPoint["e"]=e2
    dataPoint["convF"]=convF
    dataPoint["deltaF"]=deltaF
    if model==:SquareIsing
        save_object(folder*"data_T=$(T)_chi=$(chi)_boundaryCondition=$(boundaryCondition).dat",dataPoint)
        save_object(folder*"net_T=$(T)_chi=$(chi)_boundaryCondition=$(boundaryCondition).dat",curNet)
    end
    if model==:FFIsing
        save_object(folder*"data_T=$(T)_chi=$(chi)_boundaryCondition=$(boundaryCondition).dat",dataPoint)
        save_object(folder*"net_T=$(T)_chi=$(chi)_boundaryCondition=$(boundaryCondition).dat",curNet)
    end
    push!(vf,f2)
    push!(ve,e2)
    push!(vm,m2)
end
# (R1,R2)=get_R1_R2([100,100],1,curNet)

# for k=1:3
#     # curNet=iterate_ctmrg(curNet)
#     @show curNet.a
#     @show curNet.b
#     @show curNet.c
#     @show curNet.d
# end