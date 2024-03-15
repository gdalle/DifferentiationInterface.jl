using ADTypes
using BenchmarkTools
using DifferentiationInterface
using LinearAlgebra

using Diffractor: Diffractor
using Enzyme: Enzyme
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using PolyesterForwardDiff: PolyesterForwardDiff
using ReverseDiff: ReverseDiff
using Zygote: Zygote, ZygoteRuleConfig

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

function (l!::Layer{<:AbstractVector,<:AbstractVector})(
    y::AbstractVector, x::Number
)::Nothing
    y .= l!.σ.(l!.w .* x .+ l!.b)
    return nothing
end

function (l::Layer{<:AbstractVector,<:Number})(x::AbstractVector)::Number
    return l.σ(dot(l.w, x) + l.b)
end

function (l::Layer{<:AbstractMatrix,<:AbstractVector})(x::AbstractVector)::AbstractVector
    return l.σ.(l.w * x .+ l.b)
end

function (l!::Layer{<:AbstractMatrix,<:AbstractVector})(
    y::AbstractVector, x::AbstractVector
)::Nothing
    mul!(y, l!.w, x)
    y .= l!.σ.(y .+ l!.b)
    return nothing
end

## Backends

all_backends = [
    AutoChainRules(ZygoteRuleConfig()),
    AutoDiffractor(),
    AutoEnzyme(Enzyme.Forward),
    AutoEnzyme(Enzyme.Reverse),
    AutoFiniteDiff(),
    AutoForwardDiff(; chunksize=2),
    AutoPolyesterForwardDiff(; chunksize=2),
    AutoReverseDiff(),
    AutoReverseDiff(; compile=true),
    AutoZygote(),
]

## Suite

function make_suite()
    SUITE = BenchmarkGroup()

    ### Scalar to scalar
    scalar_to_scalar = Layer(randn(), randn(), tanh)

    for backend in all_backends
        add_derivative_benchmarks!(SUITE, backend, scalar_to_scalar, 1, 1)
        add_pushforward_benchmarks!(SUITE, backend, scalar_to_scalar, 1, 1)
        add_pullback_benchmarks!(SUITE, backend, scalar_to_scalar, 1, 1)
    end

    ### Scalar to vector
    for m in [10]
        scalar_to_vector = Layer(randn(m), randn(m), tanh)

        for backend in all_backends
            add_multiderivative_benchmarks!(SUITE, backend, scalar_to_vector, 1, m)
            add_pushforward_benchmarks!(SUITE, backend, scalar_to_vector, 1, m)
            add_pullback_benchmarks!(SUITE, backend, scalar_to_vector, 1, m)
        end
    end

    ### Vector to scalar
    for n in [10]
        vector_to_scalar = Layer(randn(n), randn(), tanh)

        for backend in all_backends
            add_gradient_benchmarks!(SUITE, backend, vector_to_scalar, n, 1)
            add_pushforward_benchmarks!(SUITE, backend, vector_to_scalar, n, 1)
            add_pullback_benchmarks!(SUITE, backend, vector_to_scalar, n, 1)
        end
    end

    ### Vector to vector
    for (n, m) in [(10, 10)]
        vector_to_vector = Layer(randn(m, n), randn(m), tanh)

        for backend in all_backends
            add_jacobian_benchmarks!(SUITE, backend, vector_to_vector, n, m)
            add_pushforward_benchmarks!(SUITE, backend, vector_to_vector, n, m)
            add_pullback_benchmarks!(SUITE, backend, vector_to_vector, n, m)
        end
    end

    return SUITE
end

include("utils.jl")

SUITE = make_suite()
