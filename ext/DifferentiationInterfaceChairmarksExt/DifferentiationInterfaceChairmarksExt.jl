module DifferentiationInterfaceChairmarksExt

using ADTypes: AbstractADType
using Chairmarks: @be
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

function soft_test_zero(v::Number)
    if iszero(v)
        @test v == zero(v)
    else
        @test_broken v == zero(v)
    end
end

## Selector

function DT.test_allocations(
    backends::Vector{<:AbstractADType},
    operators::Vector{Symbol},
    scenarios::Vector{<:Scenario};
)
    @testset verbose = true "Allocations" begin
        @testset "$op - $(backend_string(backend))" for op in operators, backend in backends
            if op == :pushforward_allocating
                @testset "$(typeof(s))" for s in allocating(
                    vcat(scalar_scalar(scenarios), array_scalar(scenarios))
                )
                    test_allocations_pushforward_allocating(backend, s)
                end
            elseif op == :pushforward_mutating
                @testset "$(typeof(s))" for s in mutating(scenarios)
                    test_allocations_pushforward_mutating(backend, s)
                end

            elseif op == :pullback_allocating
                @testset "$(typeof(s))" for s in allocating(
                    vcat(scalar_scalar(scenarios), array_scalar(scenarios))
                )
                    test_allocations_pullback_allocating(backend, s)
                end
            elseif op == :pullback_mutating
                @testset "$(typeof(s))" for s in mutating(scenarios)
                    test_allocations_pullback_mutating(backend, s)
                end

            elseif op == :derivative_allocating
                @testset "$(typeof(s))" for s in allocating(scalar_scalar(scenarios))
                    test_allocations_derivative_allocating(backend, s)
                end

            elseif op == :multiderivative_allocating
                nothing
            elseif op == :multiderivative_mutating
                @testset "$(typeof(s))" for s in mutating(scalar_array(scenarios))
                    test_allocations_multiderivative_mutating(backend, s)
                end

            elseif op == :gradient_allocating
                @testset "$(typeof(s))" for s in allocating(array_scalar(scenarios))
                    test_allocations_gradient_allocating(backend, s)
                end

            elseif op == :jacobian_allocating
                nothing
            elseif op == :jacobian_mutating
                @testset "$(typeof(s))" for s in mutating(array_array(scenarios))
                    test_allocations_jacobian_mutating(backend, s)
                end

            elseif op == :second_derivative_allocating
                @testset "$(typeof(s))" for s in allocating(scalar_scalar(scenarios))
                    test_allocations_second_derivative_allocating(backend, s)
                end

            elseif op == :hessian_vector_product_allocating
                @testset "$(typeof(s))" for s in allocating(array_scalar(scenarios))
                    test_allocations_hessian_vector_product_allocating(backend, s)
                end
            elseif op == :hessian_allocating
                @testset "$(typeof(s))" for s in allocating(array_scalar(scenarios))
                    test_allocations_hessian_allocating(backend, s)
                end

            else
                throw(ArgumentError("Invalid operator to test: `:$op`"))
            end
        end
    end
end

## Pushforward

function test_allocations_pushforward_allocating(ba::AbstractADType, scenario::Scenario)
    isa(mode(ba), ReverseMode) && return nothing
    (; f, x, dx, dy) = deepcopy(scenario)
    extras = prepare_pushforward(ba, f, x)
    bench1 = @be zero(dy) value_and_pushforward!(_, ba, f, x, dx, extras)
    bench2 = @be zero(dy) pushforward!(_, ba, f, x, dx, extras)
    soft_test_zero(minimum(bench1).allocs)
    soft_test_zero(minimum(bench2).allocs)
    return nothing
end

function test_allocations_pushforward_mutating(ba::AbstractADType, scenario::Scenario)
    isa(mode(ba), ReverseMode) && return nothing
    (; f, x, y, dx, dy) = deepcopy(scenario)
    f! = f
    extras = prepare_pushforward(ba, f!, x, y)
    bench1 = @be (zero(y), zero(dy)) value_and_pushforward!(
        _[1], _[2], ba, f!, x, dx, extras
    )
    soft_test_zero(minimum(bench1).allocs)
    return nothing
end

## Pullback

function test_allocations_pullback_allocating(ba::AbstractADType, scenario::Scenario)
    isa(mode(ba), ForwardMode) && return nothing
    (; f, x, dx, dy) = deepcopy(scenario)
    extras = prepare_pullback(ba, f, x)
    bench1 = @be zero(dx) value_and_pullback!(_, ba, f, x, dy, extras)
    bench2 = @be zero(dx) pullback!(_, ba, f, x, dy, extras)
    soft_test_zero(minimum(bench1).allocs)
    soft_test_zero(minimum(bench2).allocs)
    return nothing
end

function test_allocations_pullback_mutating(ba::AbstractADType, scenario::Scenario)
    isa(mode(ba), ForwardMode) && return nothing
    (; f, x, y, dx, dy) = deepcopy(scenario)
    f! = f
    extras = prepare_pullback(ba, f!, x, y)
    bench1 = @be (zero(y), zero(dx)) value_and_pullback!(_[1], _[2], ba, f!, x, dy, extras)
    soft_test_zero(minimum(bench1).allocs)
    return nothing
end

## Derivative

function test_allocations_derivative_allocating(ba::AbstractADType, scenario::Scenario)
    (; f, x) = deepcopy(scenario)
    extras = prepare_derivative(ba, f, x)
    bench1 = @be value_and_derivative(ba, f, x, extras)
    soft_test_zero(minimum(bench1).allocs)
    return nothing
end

## Multiderivative

function test_allocations_multiderivative_mutating(ba::AbstractADType, scenario::Scenario)
    (; f, x, y, dy) = deepcopy(scenario)
    f! = f
    extras = prepare_multiderivative(ba, f!, x, y)
    bench1 = @be (zero(y), zero(dy)) value_and_multiderivative!(
        _[1], _[2], ba, f!, x, extras
    )
    soft_test_zero(minimum(bench1).allocs)
    return nothing
end

## Gradient

function test_allocations_gradient_allocating(ba::AbstractADType, scenario::Scenario)
    (; f, x, dx) = deepcopy(scenario)
    extras = prepare_gradient(ba, f, x)
    bench1 = @be zero(dx) value_and_gradient!(_, ba, f, x, extras)
    bench2 = @be zero(dx) gradient!(_, ba, f, x, extras)
    soft_test_zero(minimum(bench1).allocs)
    soft_test_zero(minimum(bench2).allocs)
    return nothing
end

## Jacobian

function test_allocations_jacobian_mutating(ba::AbstractADType, scenario::Scenario)
    (; f, x, y) = deepcopy(scenario)
    f! = f
    jac_template = zeros(eltype(y), length(y), length(x))
    extras = prepare_jacobian(ba, f!, x, y)
    bench1 = @be (zero(y), zero(jac_template)) value_and_jacobian!(
        _[1], _[2], ba, f!, x, extras
    )
    soft_test_zero(minimum(bench1).allocs)
    return nothing
end

## Second derivative

function test_allocations_second_derivative_allocating(
    ba::AbstractADType, scenario::Scenario
)
    (; f, x) = deepcopy(scenario)
    extras = prepare_second_derivative(ba, f, x)
    bench1 = @be value_derivative_and_second_derivative(ba, f, x, extras)
    soft_test_zero(minimum(bench1).allocs)
    return nothing
end

## Hessian-vector product

function test_allocations_hessian_vector_product_allocating(
    ba::AbstractADType, scenario::Scenario
)
    (; f, x, dx) = deepcopy(scenario)
    extras = prepare_hessian_vector_product(ba, f, x)
    bench1 = @be zero(dx) hessian_vector_product!(_, ba, f, x, dx, extras)
    soft_test_zero(minimum(bench1).allocs)
    # TODO: add gradient
    return nothing
end

## Hessian

function test_allocations_hessian_allocating(ba::AbstractADType, scenario::Scenario)
    (; f, x, y, dx) = deepcopy(scenario)
    extras = prepare_hessian(ba, f, x)
    hess_template = zeros(eltype(y), length(x), length(x))
    bench1 = @be (zero(dx), zero(hess_template)) value_gradient_and_hessian!(
        _[1], _[2], ba, f, x, extras
    )
    bench2 = @be (zero(hess_template)) hessian!(_, ba, f, x, extras)
    soft_test_zero(minimum(bench1).allocs)
    soft_test_zero(minimum(bench2).allocs)
    return nothing
end

end
