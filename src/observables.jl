"""
    average_local_observable(net::CTMEnvironment, O::Vector{ITensor},
                             r::Vector{Int}, Lx::Int, Ly::Int) -> Number

Return the ratio of a rectangular patch contraction with insertions to the
same patch with the original local layers. `r` is the upper-left boundary
corner; the interior sites are `r + [x,y]` for `x=1:Lx`, `y=1:Ly`.
Use positive dimensions and a nonzero denominator.

At each interior position the numerator inserts
`translate_operator(O[net.r_func([x,y])], r+[x,y])`. Thus `O` is selected
using patch-relative coordinates, while the denominator uses the absolute
site `net.r_func(r+[x,y])`. Use an origin aligned with the unit cell, or
reorder `O` accordingly.

Each entry of `O` is a single tensor replacing all local layers and must
include their statistical weights and external indices in reference
coordinates. All interior sites are replaced: use the contracted original
layers for sites without an insertion. Multiple insertions give their joint
expectation, with no connected-part subtraction. Large patches may be costly
and intermediate contractions are not rescaled.
"""
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


"""
    partition_function_per_unit_cell(net::CTMEnvironment, r, Lx, Ly) -> Number

Return `(x1*x2)/(x3*x4)` from four CTM contractions: a full `Lx × Ly`
patch, a corner-only loop, a horizontal strip, and a vertical strip.
`r = [x0,y0]` is the upper-left boundary corner; the full patch interior
is `r + [x,y]` for `x=1:Lx`, `y=1:Ly`.

Use positive integer dimensions spanning complete periods of the site map,
so the shortened boundary contractions have matching indices and the
boundary rescalings cancel. The result estimates the partition weight of
the chosen cell, not its logarithm or free energy. For positive classical
weights, the log partition weight per tensor is `log(result)/(Lx*Ly)`;
multiply by `-T` for free energy with `k_B=1`. Restore any removed bulk
normalization factors separately. Requires nonzero denominators.
"""
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


"""
    calc_z(net::CTMEnvironment, r) -> Number

Contract the local layers at site `r` with the eight surrounding corner and
edge tensors. Unlike the patch-observable functions, `r` is the central
site. This scalar depends on boundary normalization and is not the
normalization-canceling unit-cell partition ratio. Internal helper.
"""
function calc_z(net::CTMEnvironment,r)
    Z=get_C(net,1,r+[-1,-1])*get_T(net,1,r+[0,-1])*get_C(net,2,r+[1,-1])*get_T(net,4,r+[-1,0])*get_A(net,r)*get_T(net,2,r+[1,0])*get_C(net,4,r+[-1,1])*get_T(net,3,r+[0,1])*get_C(net,3,r+[1,1])
    return Z[]
end

"""
    calculate_horizontal_transfer_matrix(net::CTMEnvironment, Lx::Int) -> closure

Return an edge-only transfer operator spanning `Lx` columns. It contracts
upper edges at `[x,1]` and lower edges at `[x,2]`, without bulk tensors,
then relabels the two outgoing boundary indices to the input indices.

The input ITensor must connect to the open boundary legs of corners at
`[0,1]` and `[0,2]`. Choose a positive `Lx` spanning a horizontal period
so that input and output spaces match. Internal eigensolver helper.
"""
function calculate_horizontal_transfer_matrix(net::CTMEnvironment,Lx::Int)
    function transfer_operator(x)
        ind1=commonind(x,get_C(net,1,[0,1]))
        ind2=commonind(x,get_C(net,4,[0,2]))
        for r in 1:Lx
            x=*(
                x,get_T(net,1,[r,1]),get_T(net,3,[r,2])
            )
        end
        return replaceinds(x,[
            commonind(x,get_C(net,2,[Lx+1,1])),
            commonind(x,get_C(net,3,[Lx+1,2]))
        ],[
            ind1,
            ind2
        ])
    end
end


"""
    calculate_vertical_transfer_matrix(net::CTMEnvironment, Ly::Int) -> closure

Return an edge-only transfer operator spanning `Ly` rows. It contracts left
edges at `[1,y]` and right edges at `[2,y]`, without bulk tensors, then
relabels outgoing boundary indices to the input indices.

The input ITensor must connect to the open boundary legs of corners at
`[1,0]` and `[2,0]`. Choose a positive `Ly` spanning a vertical period
so that input and output spaces match. Internal eigensolver helper.
"""
function calculate_vertical_transfer_matrix(net::CTMEnvironment,Ly::Int)
    function transfer_operator(x)
        ind1=commonind(x,get_C(net,1,[1,0]))
        ind2=commonind(x,get_C(net,2,[2,0]))
        for r in 1:Ly
            x=*(
                x,get_T(net,4,[1,r]),get_T(net,2,[2,r])
            )
        end
        return replaceinds(x,[
            commonind(x,get_C(net,4,[1,Ly+1])),
            commonind(x,get_C(net,3,[2,Ly+1]))
        ],[
            ind1,
            ind2
        ])
    end
end

"""
    transfer_matrix_eigenvalues(net::CTMEnvironment, Lx::Int, Ly::Int;
                                use_gpu=true) -> (values_x, values_y)

Request five largest-magnitude eigenvalues of each edge-only transfer map
with KrylovKit `eigsolve`. The maps span `Lx` columns and `Ly` rows;
choose positive lengths covering full periods of the site map. They contain
opposite edge pairs and no bulk tensors.

Returns unnormalized eigenvalue vectors only; eigenvectors and convergence
information are discarded. The request of five modes is fixed, and small
spaces or nonconvergence can yield fewer values or solver warnings.
Check returned lengths before taking eigenvalue ratios.

Random starting tensors use GPU storage only when
`use_gpu && CUDA.functional()`; match this setting to the environment's
storage. Existing tensors are not moved between devices.

A spectral length `-1/log(abs(values[2]/values[1]))` is in units of the full
transfer step. Multiply by `Lx` or `Ly` for lattice-spacing units when
interpreting that map as the physical transfer operator.
"""
function transfer_matrix_eigenvalues(net::CTMEnvironment,Lx::Int,Ly::Int;use_gpu=true)
    Tx=calculate_horizontal_transfer_matrix(net,Lx)
    if use_gpu && CUDA.functional()
        x0=randomITensor(noncommoninds(get_C(net,1,[0,1]),get_C(net,4,[0,2])))|>to_gpu
    else
        x0=randomITensor(noncommoninds(get_C(net,1,[0,1]),get_C(net,4,[0,2])))
    end
    valsX, vecs, info = eigsolve(Tx,x0,5,:LM)
    Ty=calculate_vertical_transfer_matrix(net,Ly)
    if use_gpu && CUDA.functional()
        x0=randomITensor(noncommoninds(get_C(net,1,[1,0]),get_C(net,2,[2,0])))|>to_gpu
    else
        x0=randomITensor(noncommoninds(get_C(net,1,[1,0]),get_C(net,2,[2,0])))
    end
    valsY, vecs, info = eigsolve(Ty,x0,5,:LM)
    return (valsX,valsY)
end
