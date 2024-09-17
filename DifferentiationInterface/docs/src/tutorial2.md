# Sparsity

We present sparsity handling with DifferentiationInterface.jl.

```@example tuto2
using BenchmarkTools
using DifferentiationInterface
import ForwardDiff, Zygote
```

Sparse AD is very useful when Jacobian or Hessian matrices have a lot of zeros.
So let us write functions that satisfy this property.

```@example tuto2
f_sparse_vector(x::AbstractVector) = diff(x .^ 2) + diff(reverse(x .^ 2))
f_sparse_scalar(x::AbstractVector) = sum(f_sparse_vector(x) .^ 2)
nothing  # hide
```

Let's also pick a random test vector.

```@example tuto2
x = float.(1:8);
```

## Dense backends

When we use the [`jacobian`](@ref) or [`hessian`](@ref) operator with a dense backend, we get a dense matrix with plenty of zeros.

```@example tuto2
dense_first_order_backend = AutoForwardDiff()
J_dense = jacobian(f_sparse_vector, dense_first_order_backend, x)
```

```@example tuto2
dense_second_order_backend = SecondOrder(AutoForwardDiff(), AutoZygote())
H_dense = hessian(f_sparse_scalar, dense_second_order_backend, x)
```

The results are correct but the procedure is very slow.
By using a sparse backend, we can get the runtime to increase with the number of nonzero elements, instead of the total number of elements.

## Sparse backends

Recipe to create a sparse backend: combine a dense backend, a sparsity detector and a compatible coloring algorithm inside [`AutoSparse`](@extref ADTypes.AutoSparse).
The following are reasonable defaults:

```@example tuto2
using SparseConnectivityTracer: TracerSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm

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

```@example tuto2
jacobian(f_sparse_vector, sparse_first_order_backend, x)
```

```@example tuto2
hessian(f_sparse_scalar, sparse_second_order_backend, x)
```

## Sparse preparation

In the examples above, we didn't use preparation.
Sparse preparation is more costly than dense preparation, but it is even more essential.
Indeed, once preparation is done, sparse differentiation is much faster than dense differentiation, because it makes fewer calls to the underlying function.
The speedup becomes very visible in large dimensions.

```@example tuto2
n = 1000
jac_extras_dense = prepare_jacobian(f_sparse_vector, dense_first_order_backend, zeros(n))
jac_extras_sparse = prepare_jacobian(f_sparse_vector, sparse_first_order_backend, zeros(n))
nothing  # hide
```

```@example tuto2
@benchmark jacobian($f_sparse_vector, $jac_extras_dense, $dense_first_order_backend, $(randn(n)))
```

```@example tuto2
@benchmark jacobian($f_sparse_vector, $jac_extras_sparse, $sparse_first_order_backend, $(randn(n)))
```
