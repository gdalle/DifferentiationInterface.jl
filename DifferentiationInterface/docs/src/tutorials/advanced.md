# Advanced tutorial

We present contexts and sparsity handling with DifferentiationInterface.jl.

```@example tuto_advanced
using BenchmarkTools
using DifferentiationInterface
import ForwardDiff, Zygote
using SparseConnectivityTracer: TracerSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm
```

## Contexts

Assume you want differentiate a multi-argument function with respect to the first argument.

```@example tuto_advanced
f_multiarg(x, c) = c * sum(abs2, x)
nothing  # hide
```

The first way, which works with every backend, is to create a closure:

```@example tuto_advanced
f_singlearg(c) = x -> f_multiarg(x, c)
nothing  # hide
```

Let's see it in action:

```@example tuto_advanced
backend = AutoForwardDiff()
x = float.(1:3)

gradient(f_singlearg(10), backend, x)
```

However, for performance reasons, it is sometimes preferrable to avoid closures and pass all arguments to the original function.
We can do this by wrapping `c` into a [`Constant`](@ref) and giving this constant to the `gradient` operator.

```@example tuto_advanced
gradient(f_multiarg, backend, x, Constant(10))
```

Preparation also works in this case, even if the constant changes before execution:

```@example tuto_advanced
prep_other_constant = prepare_gradient(f_multiarg, backend, x, Constant(-1))
gradient(f_multiarg, prep_other_constant, backend, x, Constant(10))
```

## Sparsity

Sparse AD is very useful when Jacobian or Hessian matrices have a lot of zeros.
So let us write functions that satisfy this property.

```@example tuto_advanced
f_sparse_vector(x::AbstractVector) = diff(x .^ 2) + diff(reverse(x .^ 2))
f_sparse_scalar(x::AbstractVector) = sum(f_sparse_vector(x) .^ 2)
nothing  # hide
```

### Dense backends

When we use the [`jacobian`](@ref) or [`hessian`](@ref) operator with a dense backend, we get a dense matrix with plenty of zeros.

```@example tuto_advanced
x = float.(1:8);
```

```@example tuto_advanced
dense_first_order_backend = AutoForwardDiff()
J_dense = jacobian(f_sparse_vector, dense_first_order_backend, x)
```

```@example tuto_advanced
dense_second_order_backend = SecondOrder(AutoForwardDiff(), AutoZygote())
H_dense = hessian(f_sparse_scalar, dense_second_order_backend, x)
```

The results are correct but the procedure is very slow.
By using a sparse backend, we can get the runtime to increase with the number of nonzero elements, instead of the total number of elements.

### Sparse backends

Recipe to create a sparse backend: combine a dense backend, a sparsity detector and a compatible coloring algorithm inside [`AutoSparse`](@extref ADTypes.AutoSparse).
The following are reasonable defaults:

```@example tuto_advanced
sparse_first_order_backend = AutoSparse(
    AutoForwardDiff();
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)

sparse_second_order_backend = AutoSparse(
    SecondOrder(AutoForwardDiff(), AutoZygote());
    sparsity_detector=TracerSparsityDetector(),
    coloring_algorithm=GreedyColoringAlgorithm(),
)
nothing  # hide
```

Now the resulting matrices are sparse:

```@example tuto_advanced
jacobian(f_sparse_vector, sparse_first_order_backend, x)
```

```@example tuto_advanced
hessian(f_sparse_scalar, sparse_second_order_backend, x)
```

### Sparse preparation

In the examples above, we didn't use preparation.
Sparse preparation is more costly than dense preparation, but it is even more essential.
Indeed, once preparation is done, sparse differentiation is much faster than dense differentiation, because it makes fewer calls to the underlying function.
The speedup becomes very visible in large dimensions.

```@example tuto_advanced
n = 1000
jac_prep_dense = prepare_jacobian(f_sparse_vector, dense_first_order_backend, zeros(n))
jac_prep_sparse = prepare_jacobian(f_sparse_vector, sparse_first_order_backend, zeros(n))
nothing  # hide
```

```@example tuto_advanced
@benchmark jacobian($f_sparse_vector, $jac_prep_dense, $dense_first_order_backend, $(randn(n)))
```

```@example tuto_advanced
@benchmark jacobian($f_sparse_vector, $jac_prep_sparse, $sparse_first_order_backend, $(randn(n)))
```
