module DifferentiationInterfaceChairmarksExt

using ADTypes:
    AbstractADType,
    AbstractFiniteDifferencesMode,
    AbstractForwardMode,
    AbstractReverseMode,
    AbstractSymbolicDifferentiationMode
using Chairmarks: @be, Benchmark, Sample
using DifferentiationInterface
using DifferentiationInterface:
    mode,
    supports_mutation,
    supports_pushforward,
    supports_pullback,
    supports_hvp
using DifferentiationInterface.DifferentiationTest
import DifferentiationInterface.DifferentiationTest as DT
using Test: @testset, @test

function DT.run_benchmark(
    backends::Vector{<:AbstractADType},
    operators::Vector{Symbol},
    scenarios::Vector{<:Scenario};
    allocations=false,
)
    data = BenchmarkData()
    @testset verbose = true "Allocations" begin
        @testset verbose = true "$(backend_string(backend))" for backend in backends
            @testset "$op" for op in operators
                if op == :pushforward_allocating
                    @testset "$s" for s in allocating(scenarios)
                        benchmark_pushforward_allocating!(data, backend, s; allocations)
                    end
                elseif op == :pushforward_mutating
                    @testset "$s" for s in mutating(scenarios)
                        benchmark_pushforward_mutating!(data, backend, s; allocations)
                    end

                elseif op == :pullback_allocating
                    @testset "$s" for s in allocating(scenarios)
                        benchmark_pullback_allocating!(data, backend, s; allocations)
                    end
                elseif op == :pullback_mutating
                    @testset "$s" for s in mutating(scenarios)
                        benchmark_pullback_mutating!(data, backend, s; allocations)
                    end

                elseif op == :derivative_allocating
                    @testset "$s" for s in allocating(scalar_array(scenarios))
                        benchmark_derivative_allocating!(data, backend, s; allocations)
                    end
                elseif op == :derivative_mutating
                    @testset "$s" for s in mutating(scalar_array(scenarios))
                        benchmark_derivative_mutating!(data, backend, s; allocations)
                    end

                elseif op == :gradient_allocating
                    @testset "$s" for s in allocating(array_scalar(scenarios))
                        benchmark_gradient_allocating!(data, backend, s; allocations)
                    end

                elseif op == :jacobian_allocating
                    @testset "$s" for s in allocating(array_array(scenarios))
                        benchmark_jacobian_allocating!(data, backend, s; allocations)
                    end
                elseif op == :jacobian_mutating
                    @testset "$s" for s in mutating(array_array(scenarios))
                        benchmark_jacobian_mutating!(data, backend, s; allocations)
                    end

                else
                    throw(ArgumentError("Invalid operator to benchmark: `:$op`"))
                end
            end
        end
    end
    return data
end

## Pushforward

function benchmark_pushforward_allocating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    Bool(supports_pushforward(ba)) || return nothing
    (; f, x, dx, dy) = deepcopy(scen)

    extras = prepare_pushforward(ba, f, x)
    bench1 = @be zero(dy) value_and_pushforward!(_, ba, f, x, dx, extras)
    bench2 = @be zero(dy) pushforward!(_, ba, f, x, dx, extras)
    if allocations && dy isa Number
        @test 0 == minimum(bench1).allocs
        @test 0 == minimum(bench2).allocs
    end
    record!(data, ba, scen, :value_and_pushforward!, bench1)
    record!(data, ba, scen, :pushforward!, bench2)
    return nothing
end

function benchmark_pushforward_mutating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    Bool(supports_pushforward(ba)) || return nothing
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(ba, f!, x, y)
    bench1 = @be (zero(y), zero(dy)) value_and_pushforward!(
        _[1], _[2], ba, f!, x, dx, extras
    )
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, scen, :value_and_pushforward!, bench1)
    return nothing
end

## Pullback

function benchmark_pullback_allocating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    Bool(supports_pullback(ba)) || return nothing
    (; f, x, dx, dy) = deepcopy(scen)
    extras = prepare_pullback(ba, f, x)
    bench1 = @be zero(dx) value_and_pullback!(_, ba, f, x, dy, extras)
    bench2 = @be zero(dx) pullback!(_, ba, f, x, dy, extras)
    if allocations && dy isa Number
        @test 0 == minimum(bench1).allocs
        @test 0 == minimum(bench2).allocs
    end
    record!(data, ba, scen, :value_and_pullback!, bench1)
    record!(data, ba, scen, :pullback!, bench2)
    return nothing
end

function benchmark_pullback_mutating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    Bool(supports_pullback(ba)) || return nothing
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pullback(ba, f!, x, y)
    bench1 = @be (zero(y), zero(dx)) value_and_pullback!(_[1], _[2], ba, f!, x, dy, extras)
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, scen, :value_and_pullback!, bench1)
    return nothing
end

## Derivative

function benchmark_derivative_allocating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    (; f, x, dy) = deepcopy(scen)
    extras = prepare_derivative(ba, f, x)
    bench1 = @be zero(dy) value_and_derivative!(_, ba, f, x, extras)
    # never test allocations
    record!(data, ba, scen, :value_and_derivative!, bench1)
    return nothing
end

function benchmark_derivative_mutating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dy) = deepcopy(scen)
    f! = f
    extras = prepare_derivative(ba, f!, x, y)
    bench1 = @be (zero(y), zero(dy)) value_and_derivative!(
        _[1], _[2], ba, f!, x, extras
    )
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, scen, :value_and_derivative!, bench1)
    return nothing
end

## Gradient

function benchmark_gradient_allocating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    (; f, x, dx) = deepcopy(scen)
    extras = prepare_gradient(ba, f, x)
    bench1 = @be zero(dx) value_and_gradient!(_, ba, f, x, extras)
    bench2 = @be zero(dx) gradient!(_, ba, f, x, extras)
    if allocations
        @test 0 == minimum(bench1).allocs
        @test 0 == minimum(bench2).allocs
    end
    record!(data, ba, scen, :value_and_gradient!, bench1)
    record!(data, ba, scen, :gradient!, bench2)
    return nothing
end

## Jacobian

function benchmark_jacobian_allocating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    (; f, x, y) = deepcopy(scen)
    jac_template = zeros(eltype(y), length(y), length(x))
    extras = prepare_jacobian(ba, f, x)
    bench1 = @be zero(jac_template) value_and_jacobian!(_, ba, f, x, extras)
    # never test allocations
    record!(data, ba, scen, :value_and_jacobian!, bench1)
    return nothing
end

function benchmark_jacobian_mutating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y) = deepcopy(scen)
    f! = f
    jac_template = zeros(eltype(y), length(y), length(x))
    extras = prepare_jacobian(ba, f!, x, y)
    bench1 = @be (zero(y), zero(jac_template)) value_and_jacobian!(
        _[1], _[2], ba, f!, x, extras
    )
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, scen, :value_and_jacobian!, bench1)
    return nothing
end

end
