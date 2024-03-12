using ADTypes
using ADTypes: AbstractADType
using BenchmarkTools
using DifferentiationInterface
using DifferentiationInterface:
    CustomImplem, FallbackImplem, ForwardMode, ReverseMode, autodiff_mode, handles_types

const NO_EXTRAS = nothing

## Pretty printing

pretty(::AutoChainRules{<:ZygoteRuleConfig}) = "ChainRules{Zygote}"
pretty(::AutoDiffractor) = "Diffractor (forward)"
pretty(::AutoEnzyme{Val{:forward}}) = "Enzyme (forward)"
pretty(::AutoEnzyme{Val{:reverse}}) = "Enzyme (reverse)"
pretty(::AutoFiniteDiff) = "FiniteDiff"
pretty(::AutoForwardDiff) = "ForwardDiff"
pretty(::AutoPolyesterForwardDiff) = "PolyesterForwardDiff"
pretty(::AutoReverseDiff) = "ReverseDiff"
pretty(::AutoZygote) = "Zygote"

pretty(::CustomImplem) = "custom"
pretty(::FallbackImplem) = "fallback"

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

    suite["value_and_pushforward"][(n, m)]["$(pretty(backend))"] = @benchmarkable begin
        value_and_pushforward($backend, $f, $x, $dx)
    end
    suite["value_and_pushforward!"][(n, m)]["$(pretty(backend))"] = @benchmarkable begin
        value_and_pushforward!($dy, $backend, $f, $x, $dx)
    end

    suite["pushforward"][(n, m)]["$(pretty(backend))"] = @benchmarkable begin
        pushforward($backend, $f, $x, $dx)
    end
    suite["pushforward!"][(n, m)]["$(pretty(backend))"] = @benchmarkable begin
        pushforward!($dy, $backend, $f, $x, $dx)
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

    suite["value_and_pullback"][(n, m)]["$(pretty(backend))"] = @benchmarkable begin
        value_and_pullback($backend, $f, $x, $dy)
    end
    suite["value_and_pullback!"][(n, m)]["$(pretty(backend))"] = @benchmarkable begin
        value_and_pullback!($dx, $backend, $f, $x, $dy)
    end

    suite["pullback"][(n, m)]["$(pretty(backend))"] = @benchmarkable begin
        pullback($backend, $f, $x, $dy)
    end
    suite["pullback!"][(n, m)]["$(pretty(backend))"] = @benchmarkable begin
        pullback!($dx, $backend, $f, $x, $dy)
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

    for implem in (CustomImplem(), FallbackImplem())
        backend_implem = "$(pretty(backend)) - $(pretty(implem))"

        suite["value_and_derivative"][(1, 1)][backend_implem] = @benchmarkable begin
            value_and_derivative($backend, $f, $x, $NO_EXTRAS, $implem)
        end

        suite["derivative"][(1, 1)][backend_implem] = @benchmarkable begin
            derivative($backend, $f, $x, $NO_EXTRAS, $implem)
        end
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

    for implem in (CustomImplem(), FallbackImplem())
        backend_implem = "$(pretty(backend)) - $(pretty(implem))"

        suite["value_and_multiderivative"][(1, m)][backend_implem] = @benchmarkable begin
            value_and_multiderivative($backend, $f, $x, $NO_EXTRAS, $implem)
        end
        suite["value_and_multiderivative!"][(1, m)][backend_implem] = @benchmarkable begin
            value_and_multiderivative!($multider, $backend, $f, $x, $NO_EXTRAS, $implem)
        end

        suite["multiderivative"][(1, m)][backend_implem] = @benchmarkable begin
            multiderivative($backend, $f, $x, $NO_EXTRAS, $implem)
        end
        suite["multiderivative!"][(1, m)][backend_implem] = @benchmarkable begin
            multiderivative!($multider, $backend, $f, $x, $NO_EXTRAS, $implem)
        end
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

    for implem in (CustomImplem(), FallbackImplem())
        backend_implem = "$(pretty(backend)) - $(pretty(implem))"

        suite["value_and_gradient"][(n, 1)][backend_implem] = @benchmarkable begin
            value_and_gradient($backend, $f, $x, $NO_EXTRAS, $implem)
        end
        suite["value_and_gradient!"][(n, 1)][backend_implem] = @benchmarkable begin
            value_and_gradient!($grad, $backend, $f, $x, $NO_EXTRAS, $implem)
        end

        suite["gradient"][(n, 1)][backend_implem] = @benchmarkable begin
            gradient($backend, $f, $x, $NO_EXTRAS, $implem)
        end
        suite["gradient!"][(n, 1)][backend_implem] = @benchmarkable begin
            gradient!($grad, $backend, $f, $x, $NO_EXTRAS, $implem)
        end
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

    for implem in (CustomImplem(), FallbackImplem())
        backend_implem = "$(pretty(backend)) - $(pretty(implem))"

        suite["value_and_jacobian"][(n, m)][backend_implem] = @benchmarkable begin
            value_and_jacobian($backend, $f, $x, $NO_EXTRAS, $implem)
        end
        suite["value_and_jacobian!"][(n, m)][backend_implem] = @benchmarkable begin
            value_and_jacobian!($jac, $backend, $f, $x, $NO_EXTRAS, $implem)
        end

        suite["jacobian"][(n, m)][backend_implem] = @benchmarkable begin
            jacobian($backend, $f, $x, $NO_EXTRAS, $implem)
        end
        suite["jacobian!"][(n, m)][backend_implem] = @benchmarkable begin
            jacobian!($jac, $backend, $f, $x, $NO_EXTRAS, $implem)
        end
    end

    return nothing
end
