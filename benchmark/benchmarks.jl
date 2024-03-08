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

all_custom_backends = vcat(forward_custom_backends, reverse_custom_backends)
all_fallback_backends = vcat(forward_fallback_backends, reverse_fallback_backends)
all_backends = vcat(all_custom_backends, all_fallback_backends)

## Suite

SUITE = BenchmarkGroup()

### Scalar to scalar

scalar_to_scalar = Layer(randn(), randn(), tanh)

for backend in all_backends
    handles_types(backend, Number, Number) || continue
    SUITE["value_and_derivative"][(1, 1)][string(backend)] = @benchmarkable begin
        value_and_derivative($backend, $scalar_to_scalar, x)
    end setup = (x = randn())
end

for backend in all_fallback_backends
    handles_types(backend, Number, Number) || continue
    if autodiff_mode(backend) == :forward
        SUITE["value_and_pushforward"][(1, 1)][string(backend)] = @benchmarkable begin
            value_and_pushforward($backend, $scalar_to_scalar, x, dx)
        end setup = (x = randn(); dx = randn())
    else
        SUITE["value_and_pullback"][(1, 1)][string(backend)] = @benchmarkable begin
            value_and_pullback($backend, $scalar_to_scalar, x, dy)
        end setup = (x = randn(); dy = randn())
    end
end

### Scalar to vector

for m in [10]
    scalar_to_vector = Layer(randn(m), randn(m), tanh)

    for backend in all_backends
        handles_types(backend, Number, Vector) || continue
        SUITE["value_and_multiderivative"][(1, m)][string(backend)] = @benchmarkable begin
            value_and_multiderivative($backend, $scalar_to_vector, x)
        end setup = (x = randn())
        SUITE["value_and_multiderivative!"][(1, m)][string(backend)] = @benchmarkable begin
            value_and_multiderivative!(multider, $backend, $scalar_to_vector, x)
        end setup = (x = randn(); multider = zeros($m))
    end

    for backend in all_fallback_backends
        handles_types(backend, Number, Vector) || continue
        if autodiff_mode(backend) == :forward
            SUITE["value_and_pushforward"][(1, m)][string(backend)] = @benchmarkable begin
                value_and_pushforward($backend, $scalar_to_vector, x, dx)
            end setup = (x = randn(); dx = randn())
            SUITE["value_and_pushforward!"][(1, m)][string(backend)] = @benchmarkable begin
                value_and_pushforward!(dy, $backend, $scalar_to_vector, x, dx)
            end setup = (x = randn(); dx = randn(); dy = zeros($m))
        else
            SUITE["value_and_pullback"][(1, m)][string(backend)] = @benchmarkable begin
                value_and_pullback($backend, $scalar_to_vector, x, dy)
            end setup = (x = randn(); dy = ones($m))
            SUITE["value_and_pullback!"][(1, m)][string(backend)] = @benchmarkable begin
                value_and_pullback!(dx, $backend, $scalar_to_vector, x, dy)
            end setup = (x = randn(); dy = ones($m); dx = 0.0)
        end
    end
end

### Vector to scalar

for n in [10]
    vector_to_scalar = Layer(randn(n), randn(), tanh)

    for backend in all_backends
        handles_types(backend, Vector, Number) || continue
        SUITE["value_and_gradient"][(n, 1)][string(backend)] = @benchmarkable begin
            value_and_gradient($backend, $vector_to_scalar, x)
        end setup = (x = randn($n))
        SUITE["value_and_gradient!"][(n, 1)][string(backend)] = @benchmarkable begin
            value_and_gradient!(grad, $backend, $vector_to_scalar, x)
        end setup = (x = randn($n); grad = zeros($n))
    end

    for backend in all_fallback_backends
        handles_types(backend, Vector, Number) || continue
        if autodiff_mode(backend) == :forward
            SUITE["value_and_pushforward"][(n, 1)][string(backend)] = @benchmarkable begin
                value_and_pushforward($backend, $vector_to_scalar, x, dx)
            end setup = (x = randn($n); dx = randn($n))
            SUITE["value_and_pushforward!"][(n, 1)][string(backend)] = @benchmarkable begin
                value_and_pushforward!(dy, $backend, $vector_to_scalar, x, dx)
            end setup = (x = randn($n); dx = randn($n); dy = 0.0)
        else
            SUITE["value_and_pullback"][(n, 1)][string(backend)] = @benchmarkable begin
                value_and_pullback($backend, $vector_to_scalar, x, dy)
            end setup = (x = randn($n); dy = randn())
            SUITE["value_and_pullback!"][(n, 1)][string(backend)] = @benchmarkable begin
                value_and_pullback!(dx, $backend, $vector_to_scalar, x, dy)
            end setup = (x = randn($n); dy = randn(); dx = zeros($n))
        end
    end
end

### Vector to vector

for (n, m) in [(10, 10)]
    vector_to_vector = Layer(randn(m, n), randn(m), tanh)

    for backend in all_backends
        handles_types(backend, Vector, Vector) || continue
        SUITE["value_and_jacobian"][(n, m)][string(backend)] = @benchmarkable begin
            value_and_jacobian($backend, $vector_to_vector, x)
        end setup = (x = randn($n))
        SUITE["value_and_jacobian!"][(n, m)][string(backend)] = @benchmarkable begin
            value_and_jacobian!(jac, $backend, $vector_to_vector, x)
        end setup = (x = randn($n); jac = zeros($m, $n))
    end

    for backend in all_fallback_backends
        handles_types(backend, Vector, Vector) || continue
        if autodiff_mode(backend) == :forward
            SUITE["value_and_pushforward"][(n, m)][string(backend)] = @benchmarkable begin
                value_and_pushforward($backend, $vector_to_vector, x, dx)
            end setup = (x = randn($n); dx = randn($n))
            SUITE["value_and_pushforward!"][(n, m)][string(backend)] = @benchmarkable begin
                value_and_pushforward!(dy, $backend, $vector_to_vector, x, dx)
            end setup = (x = randn($n); dx = randn($n); dy = zeros($m))
        else
            SUITE["value_and_pullback"][(n, m)][string(backend)] = @benchmarkable begin
                value_and_pullback($backend, $vector_to_vector, x, dy)
            end setup = (x = randn($n); dy = randn($m))
            SUITE["value_and_pullback!"][(n, m)][string(backend)] = @benchmarkable begin
                value_and_pullback!(dx, $backend, $vector_to_vector, x, dy)
            end setup = (x = randn($n); dy = randn($m); dx = zeros($n))
        end
    end
end

# Run benchmarks locally
# results = BenchmarkTools.run(SUITE; verbose=true)

# Compare commits locally
# using BenchmarkCI; BenchmarkCI.judge(baseline="origin/main"); BenchmarkCI.displayjudgement()
