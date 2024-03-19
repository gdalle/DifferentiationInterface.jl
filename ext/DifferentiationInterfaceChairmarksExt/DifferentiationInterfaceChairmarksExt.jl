module DifferentiationInterfaceChairmarksExt

using ADTypes: AbstractADType
using Chairmarks: @be, Benchmark, Sample
using DifferentiationInterface
using DifferentiationInterface:
    ForwardMode,
    ReverseMode,
    MutationSupported,
    MutationNotSupported,
    mode,
    mutation_behavior
using DifferentiationInterface.DifferentiationTest
import DifferentiationInterface.DifferentiationTest as DT
using Test

## Test allocations from dict of run_benchmarks

function test_allocations_aux(b::Benchmark)
    allocs = minimum(b).allocs
    if iszero(allocs)
        @test allocs == 0
    else
        @test_broken allocs == 0
    end
end

function test_allocations_aux(d::Dict)
    @testset "$k" for k in keys(d)
        test_allocations_aux(d[k])
    end
end

function DT.test_allocations(d::Dict)
    @testset verbose = true "Allocations" begin
        test_allocations_aux(d)
    end
end

## Selector

function DT.run_benchmark(
    backends::Vector{<:AbstractADType},
    operators::Vector{Symbol},
    scenarios::Vector{<:Scenario};
)
    results = Dict()
    for op in operators
        op_results = results[op] = Dict()

        for backend in backends
            local_results = op_results[backend_string(backend)] = Dict()

            if op == :pushforward_allocating
                for s in allocating(scenarios)
                    local_results[typeof(s)] = run_benchmark_pushforward_allocating(
                        backend, s
                    )
                end
            elseif op == :pushforward_mutating
                for s in mutating(scenarios)
                    local_results[typeof(s)] = run_benchmark_pushforward_mutating(
                        backend, s
                    )
                end

            elseif op == :pullback_allocating
                for s in allocating(scenarios)
                    local_results[typeof(s)] = run_benchmark_pullback_allocating(backend, s)
                end
            elseif op == :pullback_mutating
                for s in mutating(scenarios)
                    local_results[typeof(s)] = run_benchmark_pullback_mutating(backend, s)
                end

            elseif op == :derivative_allocating
                for s in allocating(scalar_scalar(scenarios))
                    local_results[typeof(s)] = run_benchmark_derivative_allocating(
                        backend, s
                    )
                end

            elseif op == :multiderivative_allocating
                for s in allocating(scalar_array(scenarios))
                    local_results[typeof(s)] = run_benchmark_multiderivative_allocating(
                        backend, s
                    )
                end
            elseif op == :multiderivative_mutating
                for s in mutating(scalar_array(scenarios))
                    local_results[typeof(s)] = run_benchmark_multiderivative_mutating(
                        backend, s
                    )
                end

            elseif op == :gradient_allocating
                for s in allocating(array_scalar(scenarios))
                    local_results[typeof(s)] = run_benchmark_gradient_allocating(backend, s)
                end

            elseif op == :jacobian_allocating
                for s in allocating(array_array(scenarios))
                    local_results[typeof(s)] = run_benchmark_jacobian_allocating(backend, s)
                end
            elseif op == :jacobian_mutating
                for s in mutating(array_array(scenarios))
                    local_results[typeof(s)] = run_benchmark_jacobian_mutating(backend, s)
                end

            elseif op == :second_derivative_allocating
                for s in allocating(scalar_scalar(scenarios))
                    local_results[typeof(s)] = run_benchmark_second_derivative_allocating(
                        backend, s
                    )
                end

            elseif op == :hessian_vector_product_allocating
                for s in allocating(array_scalar(scenarios))
                    local_results[typeof(s)] = run_benchmark_hessian_vector_product_allocating(
                        backend, s
                    )
                end
            elseif op == :hessian_allocating
                for s in allocating(array_scalar(scenarios))
                    local_results[typeof(s)] = run_benchmark_hessian_allocating(backend, s)
                end

            else
                throw(ArgumentError("Invalid operator to run_benchmark: `:$op`"))
            end
        end
    end
    return results
end

## Pushforward

function run_benchmark_pushforward_allocating(ba::AbstractADType, scenario::Scenario)
    isa(mode(ba), ReverseMode) && return Dict{Symbol,Benchmark}()
    (; f, x, dx, dy) = deepcopy(scenario)
    extras = prepare_pushforward(ba, f, x)
    return Dict(
        :value_and_pushforward! => begin
            @be zero(dy) value_and_pushforward!(_, ba, f, x, dx, extras)
        end,
        :pushforward! => begin
            @be zero(dy) pushforward!(_, ba, f, x, dx, extras)
        end,
    )
end

function run_benchmark_pushforward_mutating(ba::AbstractADType, scenario::Scenario)
    isa(mode(ba), ReverseMode) && return Dict{Symbol,Benchmark}()
    (; f, x, y, dx, dy) = deepcopy(scenario)
    f! = f
    extras = prepare_pushforward(ba, f!, x, y)
    return Dict(
        :value_and_pushforward! => begin
            @be (zero(y), zero(dy)) value_and_pushforward!(_[1], _[2], ba, f!, x, dx, extras)
        end,
    )
end

## Pullback

function run_benchmark_pullback_allocating(ba::AbstractADType, scenario::Scenario)
    isa(mode(ba), ForwardMode) && return Dict{Symbol,Benchmark}()
    (; f, x, dx, dy) = deepcopy(scenario)
    extras = prepare_pullback(ba, f, x)
    return Dict(
        :value_and_pullback! => begin
            @be zero(dx) value_and_pullback!(_, ba, f, x, dy, extras)
        end,
        :pullback! => begin
            @be zero(dx) pullback!(_, ba, f, x, dy, extras)
        end,
    )
end

function run_benchmark_pullback_mutating(ba::AbstractADType, scenario::Scenario)
    isa(mode(ba), ForwardMode) && return Dict{Symbol,Benchmark}()
    (; f, x, y, dx, dy) = deepcopy(scenario)
    f! = f
    extras = prepare_pullback(ba, f!, x, y)
    return Dict(
        :value_and_pullback! => begin
            @be (zero(y), zero(dx)) value_and_pullback!(_[1], _[2], ba, f!, x, dy, extras)
        end,
    )
end

## Derivative

function run_benchmark_derivative_allocating(ba::AbstractADType, scenario::Scenario)
    (; f, x) = deepcopy(scenario)
    extras = prepare_derivative(ba, f, x)
    return Dict(:value_and_derivative => begin
        @be value_and_derivative(ba, f, x, extras)
    end)
end

## Multiderivative

function run_benchmark_multiderivative_allocating(ba::AbstractADType, scenario::Scenario)
    (; f, x, dy) = deepcopy(scenario)
    extras = prepare_multiderivative(ba, f, x)
    return Dict(
        :value_and_multiderivative! => begin
            @be zero(dy) value_and_multiderivative!(_, ba, f, x, extras)
        end,
    )
end

function run_benchmark_multiderivative_mutating(ba::AbstractADType, scenario::Scenario)
    (; f, x, y, dy) = deepcopy(scenario)
    f! = f
    extras = prepare_multiderivative(ba, f!, x, y)
    return Dict(
        :value_and_multiderivative! => begin
            @be (zero(y), zero(dy)) value_and_multiderivative!(_[1], _[2], ba, f!, x, extras)
        end,
    )
end

## Gradient

function run_benchmark_gradient_allocating(ba::AbstractADType, scenario::Scenario)
    (; f, x, dx) = deepcopy(scenario)
    extras = prepare_gradient(ba, f, x)
    return Dict(
        :value_and_gradient! => begin
            @be zero(dx) value_and_gradient!(_, ba, f, x, extras)
        end,
        :gradient! => begin
            @be zero(dx) gradient!(_, ba, f, x, extras)
        end,
    )
end

## Jacobian

function run_benchmark_jacobian_allocating(ba::AbstractADType, scenario::Scenario)
    (; f, x, y) = deepcopy(scenario)
    jac_template = zeros(eltype(y), length(y), length(x))
    extras = prepare_jacobian(ba, f, x)
    return Dict(
        :value_and_jacobian! => begin
            @be zero(jac_template) value_and_jacobian!(_, ba, f, x, extras)
        end,
    )
end

function run_benchmark_jacobian_mutating(ba::AbstractADType, scenario::Scenario)
    (; f, x, y) = deepcopy(scenario)
    f! = f
    jac_template = zeros(eltype(y), length(y), length(x))
    extras = prepare_jacobian(ba, f!, x, y)
    return Dict(
        :value_and_jacobian! => begin
            @be (zero(y), zero(jac_template)) value_and_jacobian!(
                _[1], _[2], ba, f!, x, extras
            )
        end,
    )
end

## Second derivative

function run_benchmark_second_derivative_allocating(ba::AbstractADType, scenario::Scenario)
    (; f, x) = deepcopy(scenario)
    extras = prepare_second_derivative(ba, f, x)
    return Dict(
        :value_derivative_and_second_derivative => begin
            @be value_derivative_and_second_derivative(ba, f, x, extras)
        end,
    )
end

## Hessian-vector product

function run_benchmark_hessian_vector_product_allocating(
    ba::AbstractADType, scenario::Scenario
)
    (; f, x, dx) = deepcopy(scenario)
    extras = prepare_hessian_vector_product(ba, f, x)
    return Dict(
        :hessian_vector_product! => begin
            @be zero(dx) hessian_vector_product!(_, ba, f, x, dx, extras)
        end,
    )
end

## Hessian

function run_benchmark_hessian_allocating(ba::AbstractADType, scenario::Scenario)
    (; f, x, y, dx) = deepcopy(scenario)
    extras = prepare_hessian(ba, f, x)
    hess_template = zeros(eltype(y), length(x), length(x))
    return Dict(
        :value_gradient_and_hessian! => begin
            @be (zero(dx), zero(hess_template)) value_gradient_and_hessian!(
                _[1], _[2], ba, f, x, extras
            )
        end,
        :hessian! => begin
            @be (zero(hess_template)) hessian!(_, ba, f, x, extras)
        end,
    )
end
end
