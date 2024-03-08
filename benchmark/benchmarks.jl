using BenchmarkTools
using DifferentiationInterface
using LinearAlgebra

using Enzyme: Enzyme
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using PolyesterForwardDiff: PolyesterForwardDiff
using ReverseDiff: ReverseDiff
using Zygote: Zygote

## Settings

BenchmarkTools.DEFAULT_PARAMETERS.evals = 1
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1

## Functions

struct Layer{W<:Union{Number,AbstractArray},B<:Union{Number,AbstractArray},S<:Function}
    w::W
    b::B
    σ::S
end

function (l::Layer{<:Number,<:Number})(x::Number)::Number
    return l.σ(l.w * x + l.b)
end

function (l::Layer{<:AbstractVector,<:AbstractVector})(x::Number)::AbstractVector
    return l.σ.(l.w .* x .+ l.b)
end

function (l::Layer{<:AbstractVector,<:Number})(x::AbstractVector)::Number
    return l.σ(dot(l.w, x) + l.b)
end

function (l::Layer{<:AbstractMatrix,<:AbstractVector})(x::AbstractVector)::AbstractVector
    return l.σ.(l.w * x .+ l.b)
end

## Backends

forward_custom_backends = [
    EnzymeForwardBackend(; custom=true),
    FiniteDiffBackend(; custom=true),
    ForwardDiffBackend(; custom=true),
    PolyesterForwardDiffBackend(4; custom=true),
]

forward_fallback_backends = [
    EnzymeForwardBackend(; custom=false),
    FiniteDiffBackend(; custom=false),
    ForwardDiffBackend(; custom=false),
]

reverse_custom_backends = [
    ZygoteBackend(; custom=true),
    EnzymeReverseBackend(; custom=true),
    ReverseDiffBackend(; custom=true),
]

reverse_fallback_backends = [
    ZygoteBackend(; custom=false),
    EnzymeReverseBackend(; custom=false),
    ReverseDiffBackend(; custom=false),
]

all_backends = vcat(
    forward_custom_backends,
    forward_fallback_backends,
    reverse_custom_backends,
    reverse_fallback_backends,
)

## Suite

function make_suite()
    SUITE = BenchmarkGroup()

    ### Scalar to scalar
    scalar_to_scalar = Layer(randn(), randn(), tanh)

    for backend in all_backends
        add_derivative_benchmarks!(SUITE, backend, scalar_to_scalar, 1, 1)
    end
    for backend in forward_fallback_backends
        add_pushforward_benchmarks!(SUITE, backend, scalar_to_scalar, 1, 1)
    end
    for backend in reverse_fallback_backends
        add_pullback_benchmarks!(SUITE, backend, scalar_to_scalar, 1, 1)
    end

    ### Scalar to vector
    for m in [10]
        scalar_to_vector = Layer(randn(m), randn(m), tanh)

        for backend in all_backends
            add_multiderivative_benchmarks!(SUITE, backend, scalar_to_vector, 1, m)
        end
        for backend in forward_fallback_backends
            add_pushforward_benchmarks!(SUITE, backend, scalar_to_vector, 1, m)
        end
        for backend in reverse_fallback_backends
            add_pullback_benchmarks!(SUITE, backend, scalar_to_vector, 1, m)
        end
    end

    ### Vector to scalar
    for n in [10]
        vector_to_scalar = Layer(randn(n), randn(), tanh)

        for backend in all_backends
            add_gradient_benchmarks!(SUITE, backend, vector_to_scalar, n, 1)
        end
        for backend in forward_fallback_backends
            add_pushforward_benchmarks!(SUITE, backend, vector_to_scalar, n, 1)
        end
        for backend in reverse_fallback_backends
            add_pullback_benchmarks!(SUITE, backend, vector_to_scalar, n, 1)
        end
    end

    ### Vector to vector
    for (n, m) in [(10, 10)]
        vector_to_vector = Layer(randn(m, n), randn(m), tanh)

        for backend in all_backends
            add_jacobian_benchmarks!(SUITE, backend, vector_to_vector, n, m)
        end
        for backend in forward_fallback_backends
            add_pushforward_benchmarks!(SUITE, backend, vector_to_vector, n, m)
        end
        for backend in reverse_fallback_backends
            add_pullback_benchmarks!(SUITE, backend, vector_to_vector, n, m)
        end
    end

    return SUITE
end

include("utils.jl")

SUITE = make_suite()

# Run benchmarks locally
# results = BenchmarkTools.run(SUITE; verbose=true)

# Compare commits locally
# using BenchmarkCI; BenchmarkCI.judge(baseline="origin/main"); BenchmarkCI.displayjudgement()

# Parse into dataframe
# include("dataframe.jl")
# data = parse_benchmark_results(results; path=joinpath(@__DIR__, "results.csv"))
