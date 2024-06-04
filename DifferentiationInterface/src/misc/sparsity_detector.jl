"""
    DenseSparsityDetector

Sparsity pattern detector satisfying the [detection API](https://sciml.github.io/ADTypes.jl/stable/#Sparse-AD) of [ADTypes.jl](https://github.com/SciML/ADTypes.jl).

The nonzeros in a Jacobian or Hessian are detected by computing the relevant matrix with _dense_ AD, and thresholding the entries with a given tolerance (which can be numerically inaccurate).

!!! warning
    This detector can be very slow, and should only be used if its output can be exploited multiple times to compute many sparse matrices. 

# Fields

- `backend::AbstractADType` is the dense AD backend used under the hood
- `atol::Float64` is the minimum magnitude of a matrix entry to be considered nonzero

# Constructor

    DenseSparsityDetector(backend; atol, method)

The keyword argument `method::Symbol` can be either:

- `:iterative`: compute the matrix in a sequence of matrix-vector products (memory-efficient)
- `:direct`: compute the matrix all at once (memory-hungry but sometimes faster).

Note that the constructor is type-unstable because `method` ends up being a type parameter of the `DenseSparsityDetector` object (this is not part of the API and might change).

# Example

```jldoctest
using ADTypes, DifferentiationInterface
import ForwardDiff

detector = DenseSparsityDetector(AutoForwardDiff(); atol=1e-5, method=:direct)

ADTypes.jacobian_sparsity(diff, rand(5), detector)

# output

4×5 SparseMatrixCSC{Bool, Int64} with 8 stored entries:
 1  1  ⋅  ⋅  ⋅
 ⋅  1  1  ⋅  ⋅
 ⋅  ⋅  1  1  ⋅
 ⋅  ⋅  ⋅  1  1
```
"""
struct DenseSparsityDetector{method,B} <: ADTypes.AbstractSparsityDetector
    backend::B
    atol::Float64
end

function Base.show(io::IO, detector::DenseSparsityDetector{method}) where {method}
    @compat (; backend, atol) = detector
    return print(io, "DenseSparsityDetector{:$method}($backend; atol=$atol)")
end

function DenseSparsityDetector(
    backend::AbstractADType; atol::Float64, method::Symbol=:iterative
)
    if !(method in (:iterative, :direct))
        throw(
            ArgumentError("The keyword `method` must be either `:iterative` or `:direct`.")
        )
    end
    return DenseSparsityDetector{method,typeof(backend)}(backend, atol)
end

## Direct

function ADTypes.jacobian_sparsity(f, x, detector::DenseSparsityDetector{:direct})
    @compat (; backend, atol) = detector
    J = jacobian(f, backend, x)
    return sparse(abs.(J) .> atol)
end

function ADTypes.jacobian_sparsity(f!, y, x, detector::DenseSparsityDetector{:direct})
    @compat (; backend, atol) = detector
    J = jacobian(f!, y, backend, x)
    return sparse(abs.(J) .> atol)
end

function ADTypes.hessian_sparsity(f, x, detector::DenseSparsityDetector{:direct})
    @compat (; backend, atol) = detector
    H = hessian(f, backend, x)
    return sparse(abs.(H) .> atol)
end

## Iterative

function ADTypes.jacobian_sparsity(f, x, detector::DenseSparsityDetector{:iterative})
    @compat (; backend, atol) = detector
    y = f(x)
    n, m = length(x), length(y)
    I, J = Int[], Int[]
    if pushforward_performance(backend) isa PushforwardFast
        p = similar(y)
        for (kj, j) in enumerate(CartesianIndices(x))
            pushforward!(f, p, backend, x, basis(backend, x, j))
            for ki in LinearIndices(p)
                if abs(p[ki]) > atol
                    push!(I, ki)
                    push!(J, kj)
                end
            end
        end
    else
        p = similar(x)
        for (ki, i) in enumerate(CartesianIndices(y))
            pullback!(f, p, backend, x, basis(backend, y, i))
            for kj in LinearIndices(p)
                if abs(p[kj]) > atol
                    push!(I, ki)
                    push!(J, kj)
                end
            end
        end
    end
    return sparse(I, J, ones(Bool, length(I)), m, n)
end

function ADTypes.jacobian_sparsity(f!, y, x, detector::DenseSparsityDetector{:iterative})
    @compat (; backend, atol) = detector
    n, m = length(x), length(y)
    I, J = Int[], Int[]
    if pushforward_performance(backend) isa PushforwardFast
        p = similar(y)
        for (kj, j) in enumerate(CartesianIndices(x))
            pushforward!(f!, y, p, backend, x, basis(backend, x, j))
            for ki in LinearIndices(p)
                if abs(p[ki]) > atol
                    push!(I, ki)
                    push!(J, kj)
                end
            end
        end
    else
        p = similar(x)
        for (ki, i) in enumerate(CartesianIndices(y))
            pullback!(f!, y, p, backend, x, basis(backend, y, i))
            for kj in LinearIndices(p)
                if abs(p[kj]) > atol
                    push!(I, ki)
                    push!(J, kj)
                end
            end
        end
    end
    return sparse(I, J, ones(Bool, length(I)), m, n)
end

function ADTypes.hessian_sparsity(f, x, detector::DenseSparsityDetector{:iterative})
    @compat (; backend, atol) = detector
    n = length(x)
    I, J = Int[], Int[]
    p = similar(x)
    for (kj, j) in enumerate(CartesianIndices(x))
        hvp!(f, p, backend, x, basis(backend, x, j))
        for ki in LinearIndices(p)
            if abs(p[ki]) > atol
                push!(I, ki)
                push!(J, kj)
            end
        end
    end
    return sparse(I, J, ones(Bool, length(I)), n, n)
end
