# Run benchmarks locally by calling:
# julia -e 'using BenchmarkCI; BenchmarkCI.judge(baseline="origin/main"); BenchmarkCI.displayjudgement()'

using BenchmarkTools
using DifferentiationInterface
using LinearAlgebra

using Enzyme: Enzyme
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using PolyesterForwardDiff: PolyesterForwardDiff
using ReverseDiff: ReverseDiff
using Zygote: Zygote

struct ScalarToVector
    n::Int
end

(f::ScalarToVector)(x::Real) = collect((1:(f.n)) .* x)

scalar_to_scalar(x::Real) = x
vector_to_scalar(x::AbstractVector{<:Real}) = dot(eachindex(x), x)
vector_to_vector(x::AbstractVector{<:Real}) = eachindex(x) .* x

forward_backends = [
    EnzymeForwardBackend(; custom=true),
    EnzymeForwardBackend(; custom=false),
    FiniteDiffBackend(; custom=true),
    FiniteDiffBackend(; custom=false),
    ForwardDiffBackend(; custom=true),
    ForwardDiffBackend(; custom=false),
]

reverse_backends = [
    ZygoteBackend(; custom=true),
    ZygoteBackend(; custom=false),
    EnzymeReverseBackend(; custom=true),
    EnzymeReverseBackend(; custom=false),
    ReverseDiffBackend(; custom=true),
    ReverseDiffBackend(; custom=false),
]

backends = vcat(forward_backends, reverse_backends)

input_dims = [10]
output_dims = [10]

SUITE = BenchmarkGroup()

for backend in backends

    ## Scalar to scalar
    if !(backend isa ReverseDiffBackend)
        SUITE["derivative"][(1, 1)][string(backend)] = @benchmarkable begin
            value_and_derivative($backend, $scalar_to_scalar, x)
        end setup = (x = 1.0) evals = 1 seconds = 1
    end

    ## Scalar to vector
    if !(
        backend isa ReverseDiffBackend ||
        backend isa EnzymeReverseBackend ||
        backend isa EnzymeForwardBackend
    )
        for m in output_dims
            scalar_to_vector = ScalarToVector(m)
            SUITE["multiderivative"][(1, m)][string(backend)] = @benchmarkable begin
                value_and_multiderivative($backend, $scalar_to_vector, x)
            end setup = (x = 1.0) evals = 1 seconds = 1
        end
    end

    ## Vector to scalar
    for n in input_dims
        SUITE["gradient"][(n, 1)][string(backend)] = @benchmarkable begin
            value_and_gradient($backend, $vector_to_scalar, x)
        end setup = (x = randn($n)) evals = 1 seconds = 1
    end

    ## Vector to vector
    if !(backend isa EnzymeReverseBackend)
        for n in input_dims, m in output_dims
            backend isa EnzymeReverseBackend && continue
            SUITE["jacobian"][(n, m)][string(backend)] = @benchmarkable begin
                value_and_jacobian($backend, $vector_to_vector, x)
            end setup = (x = randn($n)) evals = 1 seconds = 1
        end
    end
end

# results = BenchmarkTools.run(SUITE; verbose=true)
