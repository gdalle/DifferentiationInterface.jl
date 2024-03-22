module DifferentiationInterfaceChairmarksExt

using ADTypes: AbstractADType
using Chairmarks: @be, Benchmark, Sample
using DifferentiationInterface
using DifferentiationInterface:
    mode, myzero, supports_mutation, supports_pushforward, supports_pullback
using DifferentiationInterface.DifferentiationTest:
    Scenario,
    BenchmarkData,
    allocating,
    backend_string,
    mutating,
    scalar_in,
    scalar_out,
    array_array,
    record!
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
                    @testset "$s" for s in allocating(scalar_in(scenarios))
                        benchmark_derivative_allocating!(data, backend, s; allocations)
                    end
                elseif op == :derivative_mutating
                    @testset "$s" for s in mutating(scalar_in(scenarios))
                        benchmark_derivative_mutating!(data, backend, s; allocations)
                    end

                elseif op == :gradient_allocating
                    @testset "$s" for s in allocating(scalar_out(scenarios))
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

    bench1 = @be myzero(dy) value_and_pushforward!(f, _, ba, x, dx)
    if allocations && dy isa Number
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, scen, :value_and_pushforward!, bench1)
    return nothing
end

function benchmark_pushforward_mutating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    Bool(supports_pushforward(ba)) || return nothing
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    bench1 = @be (myzero(y), myzero(dy)) value_and_pushforward!(f!, _[1], _[2], ba, x, dx)
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
    bench1 = @be myzero(dx) value_and_pullback!(f, _, ba, x, dy)
    if allocations && dy isa Number
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, scen, :value_and_pullback!, bench1)
    return nothing
end

function benchmark_pullback_mutating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    Bool(supports_pullback(ba)) || return nothing
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    bench1 = @be (myzero(y), myzero(dx)) value_and_pullback!(f!, _[1], _[2], ba, x, dy)
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
    bench1 = @be myzero(dy) value_and_derivative!(f, _, ba, x)
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
    bench1 = @be (myzero(y), myzero(dy)) value_and_derivative!(f!, _[1], _[2], ba, x)
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
    bench1 = @be myzero(dx) value_and_gradient!(f, _, ba, x)
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, scen, :value_and_gradient!, bench1)
    return nothing
end

## Jacobian

function benchmark_jacobian_allocating!(
    data::BenchmarkData, ba::AbstractADType, scen::Scenario; allocations::Bool
)
    (; f, x, y) = deepcopy(scen)
    jac_template = zeros(eltype(y), length(y), length(x))
    bench1 = @be myzero(jac_template) value_and_jacobian!(f, _, ba, x)
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
    bench1 = @be (myzero(y), myzero(jac_template)) value_and_jacobian!(
        f!, _[1], _[2], ba, x
    )
    if allocations
        @test 0 == minimum(bench1).allocs
    end
    record!(data, ba, scen, :value_and_jacobian!, bench1)
    return nothing
end

end
