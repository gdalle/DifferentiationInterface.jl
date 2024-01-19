# Run benchmarks locally by calling:
# julia -e 'using BenchmarkCI; BenchmarkCI.judge(baseline="origin/main"); BenchmarkCI.displayjudgement()'

using Base: Fix2
using BenchmarkTools
using DifferentiationInterface
using LinearAlgebra

using Diffractor: Diffractor
using Enzyme: Enzyme
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Zygote: Zygote

scalar_to_scalar(x::Real) = x
scalar_to_vector(x::Real, n) = collect((1:n) .* x)
vector_to_scalar(x::AbstractVector{<:Real}) = dot(1:length(x), x)
vector_to_vector(x::AbstractVector{<:Real}) = (1:length(x)) .* x

forward_backends = [EnzymeForwardBackend(), FiniteDiffBackend(), ForwardDiffBackend()]

reverse_backends = [
    ChainRulesReverseBackend(Zygote.ZygoteRuleConfig()),
    EnzymeReverseBackend(),
    ReverseDiffBackend(),
]

n_values = [10]

SUITE = BenchmarkGroup()

for n in n_values
    for backend in forward_backends
        SUITE["forward"]["scalar_to_scalar"][n][string(backend)] = @benchmarkable begin
            pushforward!(dy, $backend, scalar_to_scalar, x, dx)
        end setup = (x = 1.0; dx = 1.0; dy = 0.0) evals = 1
        if backend != EnzymeForwardBackend()  # type instability?
            SUITE["forward"]["scalar_to_vector"][n][string(backend)] = @benchmarkable begin
                pushforward!(dy, $backend, Fix2(scalar_to_vector, $n), x, dx)
            end setup = (x = 1.0; dx = 1.0; dy = zeros($n)) evals = 1
        end
        SUITE["forward"]["vector_to_vector"][n][string(backend)] = @benchmarkable begin
            pushforward!(dy, $backend, vector_to_vector, x, dx)
        end setup = (x = randn($n); dx = randn($n); dy = zeros($n)) evals = 1
    end

    for backend in reverse_backends
        if backend != ReverseDiffBackend()
            SUITE["reverse"]["scalar_to_scalar"][n][string(backend)] = @benchmarkable begin
                pullback!(dx, $backend, scalar_to_scalar, x, dy)
            end setup = (x = 1.0; dy = 1.0; dx = 0.0) evals = 1
        end
        SUITE["reverse"]["vector_to_scalar"][n][string(backend)] = @benchmarkable begin
            pullback!(dx, $backend, vector_to_scalar, x, dy)
        end setup = (x = randn($n); dy = 1.0; dx = zeros($n)) evals = 1
        if backend != EnzymeReverseBackend()
            SUITE["reverse"]["vector_to_vector"][n][string(backend)] = @benchmarkable begin
                pullback!(dx, $backend, vector_to_vector, x, dy)
            end setup = (x = randn($n); dy = randn($n); dx = zeros($n)) evals = 1
        end
    end
end
