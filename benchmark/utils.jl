using ADTypes
using ADTypes: AbstractADType
using BenchmarkTools
using DifferentiationInterface
using DifferentiationInterface: ForwardMode, ReverseMode, autodiff_mode

## Pretty printing

pretty_backend(::AutoChainRules{<:ZygoteRuleConfig}) = "ChainRules{Zygote}"
pretty_backend(::AutoDiffractor) = "Diffractor (forward)"

function pretty_backend(backend::AutoEnzyme)
    return autodiff_mode(backend) isa ForwardMode ? "Enzyme (forward)" : "Enzyme (reverse)"
end

pretty_backend(::AutoFiniteDiff) = "FiniteDiff"
pretty_backend(::AutoForwardDiff) = "ForwardDiff"
pretty_backend(::AutoPolyesterForwardDiff) = "PolyesterForwardDiff"

function pretty_backend(backend::AutoReverseDiff)
    return backend.compile ? "ReverseDiff compiled" : "ReverseDiff"
end

pretty_backend(::AutoZygote) = "Zygote"

pretty_extras(::Nothing) = "unprepared"
pretty_extras(something) = "prepared"

## Mutation

handles_mutation(::AbstractADType) = false
handles_mutation(::AutoForwardDiff) = true
handles_mutation(::AutoEnzyme) = true
handles_mutation(::AutoPolyesterForwardDiff) = true
handles_mutation(::AutoReverseDiff) = true

## Benchmark suite

function add_pushforward_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractADType, f::F, n::Integer, m::Integer
) where {F}
    x = n == 1 ? randn() : randn(n)
    dx = n == 1 ? randn() : randn(n)
    dy = m == 1 ? 0.0 : zeros(m)

    if !isa(autodiff_mode(backend), ForwardMode)
        return nothing
    end

    for extras in unique([nothing, prepare_pushforward(backend, f, x)])
        subgroup = suite[n][m]["allocating"][pretty_backend(backend)][pretty_extras(extras)]

        subgroup["value_and_pushforward"] = @benchmarkable begin
            value_and_pushforward($backend, $f, $x, $dx, $extras)
        end
        subgroup["value_and_pushforward!"] = @benchmarkable begin
            value_and_pushforward!($dy, $backend, $f, $x, $dx, $extras)
        end

        subgroup["pushforward"] = @benchmarkable begin
            pushforward($backend, $f, $x, $dx, $extras)
        end
        subgroup["pushforward!"] = @benchmarkable begin
            pushforward!($dy, $backend, $f, $x, $dx, $extras)
        end
    end

    if dy isa AbstractArray && handles_mutation(backend)
        y = similar(dy)
        for extras in unique([nothing, prepare_pushforward(backend, f, x, y)])
            subgroup = suite[n][m]["mutating"][pretty_backend(backend)][pretty_extras(
                extras
            )]

            subgroup["value_and_pushforward!"] = @benchmarkable begin
                value_and_pushforward!($y, $dy, $backend, $f, $x, $dx, $extras)
            end
        end
    end

    return nothing
end

function add_pullback_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractADType, f::F, n::Integer, m::Integer
) where {F}
    x = n == 1 ? randn() : randn(n)
    dx = n == 1 ? 0.0 : zeros(n)
    dy = m == 1 ? randn() : randn(m)

    if !isa(autodiff_mode(backend), ReverseMode)
        return nothing
    end

    for extras in unique([nothing, prepare_pullback(backend, f, x)])
        subgroup = suite[n][m]["allocating"][pretty_backend(backend)][pretty_extras(extras)]

        subgroup["value_and_pullback"] = @benchmarkable begin
            value_and_pullback($backend, $f, $x, $dy, $extras)
        end
        subgroup["value_and_pullback!"] = @benchmarkable begin
            value_and_pullback!($dx, $backend, $f, $x, $dy, $extras)
        end

        subgroup["pullback"] = @benchmarkable begin
            pullback($backend, $f, $x, $dy, $extras)
        end
        subgroup["pullback!"] = @benchmarkable begin
            pullback!($dx, $backend, $f, $x, $dy, $extras)
        end
    end

    if dy isa AbstractArray && handles_mutation(backend)
        y = similar(dy)
        for extras in unique([nothing, prepare_pullback(backend, f, x, y)])
            subgroup = suite[n][m]["mutating"][pretty_backend(backend)][pretty_extras(
                extras
            )]

            subgroup["value_and_pullback!"] = @benchmarkable begin
                value_and_pullback!($y, $dx, $backend, $f, $x, $dy, $extras)
            end
        end
    end

    return nothing
end

function add_derivative_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractADType, f::F, n::Integer, m::Integer
) where {F}
    @assert n == m == 1

    x = randn()

    for extras in unique([nothing, prepare_derivative(backend, f, x)])
        subgroup = suite[n][m]["allocating"][pretty_backend(backend)][pretty_extras(extras)]

        subgroup["value_and_derivative"] = @benchmarkable begin
            value_and_derivative($backend, $f, $x, $extras)
        end

        subgroup["derivative"] = @benchmarkable begin
            derivative($backend, $f, $x, $extras)
        end
    end

    return nothing
end

function add_multiderivative_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractADType, f::F, n::Integer, m::Integer
) where {F}
    @assert n == 1

    x = randn()
    multider = zeros(m)

    for extras in unique([nothing, prepare_multiderivative(backend, f, x)])
        subgroup = suite[n][m]["allocating"][pretty_backend(backend)][pretty_extras(extras)]

        subgroup["value_and_multiderivative"] = @benchmarkable begin
            value_and_multiderivative($backend, $f, $x, $extras)
        end
        subgroup["value_and_multiderivative!"] = @benchmarkable begin
            value_and_multiderivative!($multider, $backend, $f, $x, $extras)
        end

        subgroup["multiderivative"] = @benchmarkable begin
            multiderivative($backend, $f, $x, $extras)
        end
        subgroup["multiderivative!"] = @benchmarkable begin
            multiderivative!($multider, $backend, $f, $x, $extras)
        end
    end

    if handles_mutation(backend)
        y = zeros(m)
        for extras in unique([nothing, prepare_multiderivative(backend, f, x, y)])
            subgroup = suite[n][m]["mutating"][pretty_backend(backend)][pretty_extras(
                extras
            )]

            subgroup["value_and_multiderivative!"] = @benchmarkable begin
                value_and_multiderivative!($y, $multider, $backend, $f, $x, $extras)
            end
        end
    end

    return nothing
end

function add_gradient_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractADType, f::F, n::Integer, m::Integer
) where {F}
    @assert m == 1

    x = randn(n)
    grad = zeros(n)

    for extras in unique([nothing, prepare_gradient(backend, f, x)])
        subgroup = suite[n][m]["allocating"][pretty_backend(backend)][pretty_extras(extras)]

        subgroup["value_and_gradient"] = @benchmarkable begin
            value_and_gradient($backend, $f, $x, $extras)
        end
        subgroup["value_and_gradient!"] = @benchmarkable begin
            value_and_gradient!($grad, $backend, $f, $x, $extras)
        end

        subgroup["gradient"] = @benchmarkable begin
            gradient($backend, $f, $x, $extras)
        end
        subgroup["gradient!"] = @benchmarkable begin
            gradient!($grad, $backend, $f, $x, $extras)
        end
    end

    return nothing
end

function add_jacobian_benchmarks!(
    suite::BenchmarkGroup, backend::AbstractADType, f::F, n::Integer, m::Integer
) where {F}
    x = randn(n)
    jac = zeros(m, n)

    for extras in unique([nothing, prepare_jacobian(backend, f, x)])
        subgroup = suite[n][m]["allocating"][pretty_backend(backend)][pretty_extras(extras)]

        subgroup["value_and_jacobian"] = @benchmarkable begin
            value_and_jacobian($backend, $f, $x, $extras)
        end
        subgroup["value_and_jacobian!"] = @benchmarkable begin
            value_and_jacobian!($jac, $backend, $f, $x, $extras)
        end

        subgroup["jacobian"] = @benchmarkable begin
            jacobian($backend, $f, $x, $extras)
        end
        subgroup["jacobian!"] = @benchmarkable begin
            jacobian!($jac, $backend, $f, $x, $extras)
        end
    end

    if handles_mutation(backend)
        y = zeros(m)
        for extras in unique([nothing, prepare_jacobian(backend, f, x, y)])
            subgroup = suite[n][m]["mutating"][pretty_backend(backend)][pretty_extras(
                extras
            )]

            subgroup["value_and_jacobian!"] = @benchmarkable begin
                value_and_jacobian!($y, $jac, $backend, $f, $x, $extras)
            end
        end
    end

    return nothing
end
