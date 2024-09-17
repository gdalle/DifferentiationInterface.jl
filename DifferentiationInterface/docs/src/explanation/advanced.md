# Advanced features

## Contexts

### Additional arguments

For all operators provided DifferentiationInterface, there can be only one differentiated (or "active") argument, which we call `x`.
However, the release v0.6 introduced the possibility of additional "context" arguments, which are not differentiated but still passed to the function after `x`.

Contexts can be useful if you have a function `y = f(x, a, b, c, ...)` or `f!(y, x, a, b, c, ...)` and you want derivatives of `y` with respect to `x` only.
Another option would be creating a closure, but that is sometimes undesirable.

!!! warning
    This feature is still experimental, and will likely not be supported by all backends.
    At the moment, it only works with ForwardDiff.

### Types of contexts

Every context argument must be wrapped in a subtype of [`Context`](@ref) and come after the differentiated input `x`.
Right now, there is only one kind of context, namely [`Constant`](@ref), but we might add more.
Semantically, calling

```julia
gradient(f, backend, x, Constant(c))
```

computes the partial gradient of `f(x, c)` with respect to `x`, while keeping `c` constant.
Importantly, one can prepare an operator with an arbitrary value `c'` of the constant (subject to the usual restrictions on preparation).

## Sparsity

When faced with sparse Jacobian or Hessian matrices, one can take advantage of their sparsity pattern to speed up the computation.
DifferentiationInterface does this automatically if you pass a backend of type [`AutoSparse`](@extref ADTypes.AutoSparse).

!!! tip
    To know more about sparse AD, read the survey [_What Color Is Your Jacobian? Graph Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711) (Gebremedhin et al., 2005).

### `AutoSparse` object

An `AutoSparse` backend must be constructed from three ingredients:

1. An underlying (dense) backend
2. A sparsity pattern detector like:
   - [`TracerSparsityDetector`](@extref SparseConnectivityTracer.TracerSparsityDetector) from [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl)
   - [`SymbolicsSparsityDetector`](@extref Symbolics.SymbolicsSparsityDetector) from [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl)
   - [`DenseSparsityDetector`](@ref) from DifferentiationInterface.jl (beware that this detector only gives a locally valid pattern)
3. A coloring algorithm: [`GreedyColoringAlgorithm`](@extref SparseMatrixColorings.GreedyColoringAlgorithm) from [SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl) is the only one we support. As a result, sparse AD is now located in a package extension which depends on SparseMatrixColorings.jl.

`AutoSparse` backends only support [`jacobian`](@ref) and [`hessian`](@ref) (as well as their variants), because other operators do not output matrices.
To obtain sparse Hessians, you need to put the `SecondOrder` backend inside `AutoSparse`, and not the other way around.

!!! note
    Symbolic backends have built-in sparsity handling, so `AutoSparse(AutoSymbolics())` and `AutoSparse(AutoFastDifferentiation())` do not need additional configuration for pattern detection or coloring.

### Cost of sparse preparation

The preparation step of `jacobian` or `hessian` with an `AutoSparse` backend can be long, because it needs to detect the sparsity pattern and perform a matrix coloring.
But after preparation, the more zeros are present in the matrix, the greater the speedup will be compared to dense differentiation.

!!! danger
    The result of preparation for an `AutoSparse` backend cannot be reused if the sparsity pattern changes.
