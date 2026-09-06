# MultiSiteCTMRG.jl

MultiSiteCTMRG contracts periodic square tensor networks with multiple
inequivalent sites using directional corner transfer matrix renormalization
group (CTMRG). Each site can contain one tensor or several layers, such as
the bra and ket layers of an iPEPS norm network.

The package builds open-boundary environments, performs normalized CTMRG
sweeps, contracts rectangular patches with observable insertions, and computes
horizontal and vertical edge-transfer spectra. Model tensors and convergence
criteria are supplied by the caller.

## Unit cells, layers, and coordinates

An environment stores:

| Field | Meaning |
|:--|:--|
| `χ` | Maximum bond dimension retained by projector SVDs |
| `A[k]` | Vector of local tensor layers for inequivalent site `k` |
| `T[θ,k]` | Edge tensor for direction `θ` and site `k` |
| `C[θ,k]` | Corner tensor for direction `θ` and site `k` |
| `r_func(r)` | Map any integer lattice coordinate `[x,y]` to its site index |
| `List_sites[k]` | Representative coordinate of site `k` |

The map must work outside the reference cell, including negative coordinates,
and satisfy `r_func(List_sites[k]) == k`. For a rectangular cell:

```julia
site_index(r) = mod1(r[1], Lx) + Lx * (mod1(r[2], Ly) - 1)
sites = [[x, y] for y in 1:Ly for x in 1:Lx]
```

All stored tensors use the same reference origin **`[1,1]`**, including
tensors for other inequivalent sites. Indices carry integer coordinate tags
such as `"x=1,y=0"`. Accessors select the site with `r_func` and translate
the stored tags by `r - [1,1]`.

For the convention used in the example, each site's right and down bonds
carry `"x=1,y=1"`, its left bond carries `"x=0,y=1"`, and its up bond
carries `"x=1,y=0"`. Neighboring tensors must also share the appropriate
index **IDs and prime levels**. Matching tags and dimensions alone are
insufficient. Build bond indices once and reuse them on both ends.

Coordinate translation replaces the full tag set of translated indices.
Use the coordinate tags alone for spatial indexing; additional descriptive
tags are not preserved. `get_A` translates every index, so even on-site
indices shared between layers need coordinate tags.

`translate_operator(O, r)` shifts by `r - [1,1]`, rather than by `r`.
To translate by displacement `d`, pass `d + [1,1]`. To bring a tensor
located at `r` back to reference coordinates, pass `[2,2] - r`.

### Boundary directions

Coordinates increase rightward and downward. Around a central site `ra`:

| Direction `θ` | Edge `T` position relative to `ra` | Corner `C` position relative to `ra` |
|:--|:--|:--|
| 1 | Up: `[0,-1]` | Upper left: `[-1,-1]` |
| 2 | Right: `[1,0]` | Upper right: `[1,-1]` |
| 3 | Down: `[0,1]` | Lower right: `[1,1]` |
| 4 | Left: `[-1,0]` | Lower left: `[-1,1]` |

`get_T(net, θ, r)` and `get_C(net, θ, r)` take the **boundary tensor's**
position. Their three-coordinate overloads, `get_T(net, r, ra)` and
`get_C(net, r, ra)`, infer the direction from `r - ra`.
`get_A(net, r)` always returns a vector, even for a single layer.

## Initialization and iteration

`initialise_CTMRG_open_BC(χ, A, r_func, List_sites; use_gpu=false)` contracts
outward legs with all-ones boundary tensors and combines remaining boundary
legs across layers. This creates a seed environment; it is not converged,
normalized, or initially truncated to `χ`.

Each layer must have a matching virtual bond to the corresponding layer at
every nearest neighbor. Shared on-site indices contract between layers.
For example, an iPEPS norm can be supplied as one bra tensor and one ket
tensor per site, provided the virtual and physical index conventions match.

Each `iterate_ctmrg(net)` call sweeps **left, up, right, down**, using projector
SVDs with `maxdim=net.χ` and `cutoff=1e-15`. New corners and edges are
normalized by their maximum absolute entries. Zero boundary tensors are not
handled by that normalization.

Always assign the returned state:

```julia
net = iterate_ctmrg(net)
```

The return value is one `CTMEnvironment`, not a tuple of state and diagnostics.
There is no convergence loop or GPU keyword on `iterate_ctmrg`. Measurements
contract the current tensors directly; no separate environment-cache refresh
or normalization call is needed.

## Examples 
### Fully Frustrated Square-Ising model
Find in `examples/fully_frustrated_square_ising.jl` an implementation of the fully-frustrated Ising model on the square lattice, as defined by [Villain 1997](https://iopscience.iop.org/article/10.1088/0022-3719/10/10/014).

### Spin-2 AKLT state
Find in `examples/spin_2_aklt.jl` an implementation of the spin-2 AKLT wavefunction on the square lattice. The IPEPS wavefunction is created and contracted using Multi-Site CTMRG. 

## Observables and partition weights

| Function | Return value |
|:--|:--|
| `partition_function_per_unit_cell(net, r, Lx, Ly)` | Partition-weight ratio for a complete periodic patch |
| `average_local_observable(net, O, r, Lx, Ly)` | Patch contraction with insertions divided by its unmodified contraction |
| `transfer_matrix_eigenvalues(net, Lx, Ly; use_gpu=false)` | Pair of unnormalized horizontal and vertical eigenvalue vectors |

For patch contractions, `r` is the **upper-left boundary corner**, and the
interior sites are `r + [x,y]`, with `x=1:Lx` and `y=1:Ly`.
Use patch dimensions covering full periods for
`partition_function_per_unit_cell`; its shortened boundary loops rely on
periodic index matching. The ratio cancels boundary normalization factors.
It returns a weight, not a logarithm or a free energy. For positive classical
weights, use `log(z)/(Lx*Ly)` per repeated tensor and restore any bulk
rescaling separately.

`O` is a `Vector{ITensor}` with one reference-coordinate insertion per
site index. Each entry replaces **all layers** at that position and includes
their weights and external indices. To leave a site unmodified, use the
contraction of its original layers, for example `*(net.A[k]...)`.

The implementation selects insertions with `O[net.r_func([x,y])]`, using
patch-relative coordinates, while selecting bulk tensors with
`net.r_func(r+[x,y])`. An origin aligned with the unit cell, such as
`r=[0,0]` in the example, makes these agree. Otherwise reorder `O` to match
the shifted patch. Multiple nontrivial insertions measure their joint
expectation; no disconnected contribution is subtracted.

### Transfer spectra

The horizontal map contracts top and bottom edge pairs over `Lx` columns;
the vertical map contracts left and right edge pairs over `Ly` rows.
Both are edge-only maps, without bulk tensors. Choose complete periods so
input and output spaces match.

The solver always requests five largest-magnitude eigenvalues in each
direction. Small invariant subspaces can yield fewer values and KrylovKit
warnings; inspect the returned lengths before indexing. Eigenvectors and
convergence information are discarded by the API.

For two suitable nonzero eigenvalues, the length
`-1/log(abs(values[2]/values[1]))` is in units of a complete transfer step.
Multiply by `Lx` or `Ly` to express it in lattice spacings when that transfer
map represents the physical correlations of interest.

## GPU storage and remembered combiners

`to_gpu(t)` and `to_cpu(t)` adapt individual ITensors, not whole environments.
For GPU initialization, move every input layer first:

```julia
# Requires a functional CUDA device. Here A, site_index, and sites are your inputs.
A_gpu = [to_gpu.(layers) for layers in A]
net_gpu = initialise_CTMRG_open_BC(χ, A_gpu, site_index, sites; use_gpu=true)
net_gpu = iterate_ctmrg(net_gpu)
```

Initialization creates GPU boundary factors only when
`use_gpu && CUDA.functional()`; it does not move the supplied bulk layers.
The eigenvalue routine similarly uses `use_gpu` only for its random starting
tensors. Keep device settings consistent with the environment's storage.
`to_gpu` itself does not fall back to the CPU.

`combiner_with_memory()` creates a closure that reuses a combined-index ID
for the same input indices across translated tensors. Its cache ignores tags
and input order, while preserving IDs and prime levels. Use a consistent
input ordering and reuse the same closure when a fused bond must be shared.

## API documentation and related reading

Julia's help mode provides signatures, conventions, and limitations:
`?CTMEnvironment`, `?translate_operator`, or `?average_local_observable`.
The active implementation is loaded from `utils.jl`, `initialise.jl`,
`iterate.jl`, and `observables.jl`. `src/old_multi_site.jl` is legacy code
and is not loaded by the module.

This implementation of Multi-Site CTMRG follows closely the method explained in Juraj Hasik's [PhD thesis](https://hdl.handle.net/20.500.11767/103941).