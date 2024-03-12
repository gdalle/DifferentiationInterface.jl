using ADTypes
using ADTypes: AbstractADType
using BenchmarkTools
using DifferentiationInterface
using DifferentiationInterface: ForwardMode, ReverseMode, autodiff_mode, handles_types

## Pretty printing

pretty_backend(::AutoChainRules{<:ZygoteRuleConfig}) = "ChainRules{Zygote}"
pretty_backend(::AutoDiffractor) = "Diffractor (forward)"
pretty_backend(::AutoEnzyme{Val{:forward}}) = "Enzyme (forward)"
pretty_backend(::AutoEnzyme{Val{:reverse}}) = "Enzyme (reverse)"
pretty_backend(::AutoFiniteDiff) = "FiniteDiff"
pretty_backend(::AutoForwardDiff) = "ForwardDiff"
pretty_backend(::AutoPolyesterForwardDiff) = "PolyesterForwardDiff"
pretty_backend(::AutoReverseDiff) = "ReverseDiff"
pretty_backend(::AutoZygote) = "Zygote"

pretty_extras(::Nothing) = "unprepared"
pretty_extras(something) = "prepared"

## Benchmark suite

function add_pushforward_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractADType, f::F, n::Integer, m::Integer
) where {F}
    x = n == 1 ? randn() : randn(n)
    dx = n == 1 ? randn() : randn(n)
    dy = m == 1 ? 0.0 : zeros(m)

    if !isa(autodiff_mode(backend), ForwardMode) ||
        !handles_types(backend, typeof(x), typeof(dy))
        return nothing
    end

    for extras in unique([nothing, prepare_pushforward(backend, f, x)])
        subgroup = suite[(n, m)][pretty_backend(backend)][pretty_extras(extras)]

        subgroup["value_and_pushforward"] = @benchmarkable begin
            value_and_pushforward($backend, $f, $x, $dx, $extras)
        end evals = 1
        subgroup["value_and_pushforward!"] = @benchmarkable begin
            value_and_pushforward!($dy, $backend, $f, $x, $dx, $extras)
        end evals = 1

        subgroup["pushforward"] = @benchmarkable begin
            pushforward($backend, $f, $x, $dx, $extras)
        end evals = 1
        subgroup["pushforward!"] = @benchmarkable begin
            pushforward!($dy, $backend, $f, $x, $dx, $extras)
        end evals = 1
    end

    return nothing
end

function add_pullback_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractADType, f::F, n::Integer, m::Integer
) where {F}
    x = n == 1 ? randn() : randn(n)
    dx = n == 1 ? 0.0 : zeros(n)
    dy = m == 1 ? randn() : randn(m)

    if !isa(autodiff_mode(backend), ReverseMode) ||
        !handles_types(backend, typeof(x), typeof(dy))
        return nothing
    end

    extras = prepare_pullback(backend, f, x)

    for extras in unique([nothing, prepare_pullback(backend, f, x)])
        subgroup = suite[(n, m)][pretty_backend(backend)][pretty_extras(extras)]

        subgroup["value_and_pullback"] = @benchmarkable begin
            value_and_pullback($backend, $f, $x, $dy, $extras)
        end evals = 1
        subgroup["value_and_pullback!"] = @benchmarkable begin
            value_and_pullback!($dx, $backend, $f, $x, $dy, $extras)
        end evals = 1

        subgroup["pullback"] = @benchmarkable begin
            pullback($backend, $f, $x, $dy, $extras)
        end evals = 1
        subgroup["pullback!"] = @benchmarkable begin
            pullback!($dx, $backend, $f, $x, $dy, $extras)
        end evals = 1
    end

    return nothing
end

function add_derivative_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractADType, f::F, n::Integer, m::Integer
) where {F}
    @assert n == m == 1
    if !handles_types(backend, Number, Number)
        return nothing
    end

    x = randn()

    for extras in unique([nothing, prepare_derivative(backend, f, x)])
        subgroup = suite[(n, m)][pretty_backend(backend)][pretty_extras(extras)]

        subgroup["value_and_derivative"] = @benchmarkable begin
            value_and_derivative($backend, $f, $x, $extras)
        end evals = 1

        subgroup["derivative"] = @benchmarkable begin
            derivative($backend, $f, $x, $extras)
        end evals = 1
    end

    return nothing
end

function add_multiderivative_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractADType, f::F, n::Integer, m::Integer
) where {F}
    @assert n == 1
    if !handles_types(backend, Number, Vector)
        return nothing
    end

    x = randn()
    multider = zeros(m)

    for extras in unique([nothing, prepare_multiderivative(backend, f, x)])
        subgroup = suite[(n, m)][pretty_backend(backend)][pretty_extras(extras)]

        subgroup["value_and_multiderivative"] = @benchmarkable begin
            value_and_multiderivative($backend, $f, $x, $extras)
        end evals = 1
        subgroup["value_and_multiderivative!"] = @benchmarkable begin
            value_and_multiderivative!($multider, $backend, $f, $x, $extras)
        end evals = 1

        subgroup["multiderivative"] = @benchmarkable begin
            multiderivative($backend, $f, $x, $extras)
        end evals = 1
        subgroup["multiderivative!"] = @benchmarkable begin
            multiderivative!($multider, $backend, $f, $x, $extras)
        end evals = 1
    end

    return nothing
end

function add_gradient_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractADType, f::F, n::Integer, m::Integer
) where {F}
    @assert m == 1
    if !handles_types(backend, Vector, Number)
        return nothing
    end

    x = randn(n)
    grad = zeros(n)

    for extras in unique([nothing, prepare_gradient(backend, f, x)])
        subgroup = suite[(n, m)][pretty_backend(backend)][pretty_extras(extras)]

        subgroup["value_and_gradient"] = @benchmarkable begin
            value_and_gradient($backend, $f, $x, $extras)
        end evals = 1
        subgroup["value_and_gradient!"] = @benchmarkable begin
            value_and_gradient!($grad, $backend, $f, $x, $extras)
        end evals = 1

        subgroup["gradient"] = @benchmarkable begin
            gradient($backend, $f, $x, $extras)
        end evals = 1
        subgroup["gradient!"] = @benchmarkable begin
            gradient!($grad, $backend, $f, $x, $extras)
        end evals = 1
    end

    return nothing
end

function add_jacobian_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractADType, f::F, n::Integer, m::Integer
) where {F}
    if !handles_types(backend, Vector, Vector)
        return nothing
    end

    x = randn(n)
    jac = zeros(m, n)

    for extras in unique([nothing, prepare_jacobian(backend, f, x)])
        subgroup = suite[(n, m)][pretty_backend(backend)][pretty_extras(extras)]

        subgroup["value_and_jacobian"] = @benchmarkable begin
            value_and_jacobian($backend, $f, $x, $extras)
        end evals = 1
        subgroup["value_and_jacobian!"] = @benchmarkable begin
            value_and_jacobian!($jac, $backend, $f, $x, $extras)
        end evals = 1

        subgroup["jacobian"] = @benchmarkable begin
            jacobian($backend, $f, $x, $extras)
        end evals = 1
        subgroup["jacobian!"] = @benchmarkable begin
            jacobian!($jac, $backend, $f, $x, $extras)
        end evals = 1
    end

    return nothing
end
