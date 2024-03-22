module DifferentiationInterfaceJETExt

using ADTypes: AbstractADType
using DifferentiationInterface
using DifferentiationInterface:
    mode, supports_mutation, supports_pushforward, supports_pullback
using DifferentiationInterface.DifferentiationTest:
    Scenario, allocating, backend_string, mutating, scalar_in, scalar_out, array_array
import DifferentiationInterface.DifferentiationTest as DT
using JET: @test_call, @test_opt
using LinearAlgebra: LinearAlgebra
using Test: @testset, @test

## Selector

function DT.test_type_stability(
    backends::Vector{<:AbstractADType},
    operators::Vector{Symbol},
    scenarios::Vector{<:Scenario};
)
    @testset verbose = true "Type stability" begin
        @testset verbose = true "$(backend_string(backend))" for backend in backends
            @testset "$op" for op in operators
                if op == :pushforward_allocating
                    @testset "$s" for s in allocating(scenarios)
                        test_type_pushforward_allocating(backend, s)
                    end
                elseif op == :pushforward_mutating
                    @testset "$s" for s in mutating(scenarios)
                        test_type_pushforward_mutating(backend, s)
                    end

                elseif op == :pullback_allocating
                    @testset "$s" for s in allocating(scenarios)
                        test_type_pullback_allocating(backend, s)
                    end
                elseif op == :pullback_mutating
                    @testset "$s" for s in mutating(scenarios)
                        test_type_pullback_mutating(backend, s)
                    end

                elseif op == :derivative_allocating
                    @testset "$s" for s in allocating(scalar_in(scenarios))
                        test_type_derivative_allocating(backend, s)
                    end
                elseif op == :derivative_mutating
                    @testset "$s" for s in mutating(scalar_in(scenarios))
                        test_type_derivative_mutating(backend, s)
                    end

                elseif op == :gradient_allocating
                    @testset "$s" for s in allocating(scalar_out(scenarios))
                        test_type_gradient_allocating(backend, s)
                    end

                elseif op == :jacobian_allocating
                    @testset "$s" for s in allocating(array_array(scenarios))
                        test_type_jacobian_allocating(backend, s)
                    end
                elseif op == :jacobian_mutating
                    @testset "$s" for s in mutating(array_array(scenarios))
                        test_type_jacobian_mutating(backend, s)
                    end

                else
                    throw(ArgumentError("Invalid operator to test: `:$op`"))
                end
            end
        end
    end
end

## Pushforward 

function test_type_pushforward_allocating(ba::AbstractADType, scen::Scenario)
    Bool(supports_pushforward(ba)) || return nothing
    (; f, x, dx, dy) = deepcopy(scen)
    extras = prepare_pushforward(f, ba, x)
    dy_in = zero(dy)

    @test_call value_and_pushforward!(f, dy_in, ba, x, dx, extras)
    @test_opt value_and_pushforward!(f, dy_in, ba, x, dx, extras)

    @test_call value_and_pushforward(f, ba, x, dx, extras)
    @test_opt value_and_pushforward(f, ba, x, dx, extras)
end

function test_type_pushforward_mutating(ba::AbstractADType, scen::Scenario)
    Bool(supports_pushforward(ba)) || return nothing
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(f!, ba, y, x)
    y_in = zero(y)
    dy_in = zero(dy)

    @test_call value_and_pushforward!(f!, y_in, dy_in, ba, x, dx, extras)
    @test_opt value_and_pushforward!(f!, y_in, dy_in, ba, x, dx, extras)
end

## Pullback

function test_type_pullback_allocating(ba::AbstractADType, scen::Scenario)
    Bool(supports_pullback(ba)) || return nothing
    (; f, x, dx, dy) = deepcopy(scen)
    extras = prepare_pullback(f, ba, x)
    dx_in = zero(dx)

    @test_call value_and_pullback!(f, dx_in, ba, x, dy, extras)
    @test_opt value_and_pullback!(f, dx_in, ba, x, dy, extras)

    @test_call value_and_pullback(f, ba, x, dy, extras)
    @test_opt value_and_pullback(f, ba, x, dy, extras)
end

function test_type_pullback_mutating(ba::AbstractADType, scen::Scenario)
    Bool(supports_pullback(ba)) || return nothing
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pullback(f!, ba, y, x)
    y_in = zero(y)
    dx_in = zero(dx)

    @test_call value_and_pullback!(f!, y_in, dx_in, ba, x, dy, extras)
    @test_opt value_and_pullback!(f!, y_in, dx_in, ba, x, dy, extras)
end

## Derivative

function test_type_derivative_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, dy) = deepcopy(scen)
    extras = prepare_derivative(f, ba, x)
    der_in = zero(dy)

    @test_call value_and_derivative!(f, der_in, ba, x, extras)
    @test_opt value_and_derivative!(f, der_in, ba, x, extras)

    @test_call value_and_derivative(f, ba, x, extras)
    @test_opt value_and_derivative(f, ba, x, extras)
end

function test_type_derivative_mutating(ba::AbstractADType, scen::Scenario)
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dy) = deepcopy(scen)
    f! = f
    extras = prepare_derivative(f!, ba, y, x)
    y_in = zero(y)
    der_in = zero(dy)

    @test_call value_and_derivative!(f!, y_in, der_in, ba, x, extras)
    @test_opt value_and_derivative!(f!, y_in, der_in, ba, x, extras)
end

## Gradient

function test_type_gradient_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, dx) = deepcopy(scen)
    extras = prepare_gradient(f, ba, x)
    grad_in = zero(dx)

    @test_call value_and_gradient!(f, grad_in, ba, x, extras)
    @test_opt value_and_gradient!(f, grad_in, ba, x, extras)

    @test_call value_and_gradient(f, ba, x, extras)
    @test_opt value_and_gradient(f, ba, x, extras)
end

## Jacobian

function test_type_jacobian_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)
    jac_in = zeros(eltype(y), length(y), length(x))

    @test_call value_and_jacobian!(f, jac_in, ba, x, extras)
    @test_opt value_and_jacobian!(f, jac_in, ba, x, extras)

    @test_call value_and_jacobian(f, ba, x, extras)
    @test_opt value_and_jacobian(f, ba, x, extras)
end

function test_type_jacobian_mutating(ba::AbstractADType, scen::Scenario)
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, ba, y, x)
    y_in = zero(y)
    jac_in = zeros(eltype(y), length(y), length(x))

    @test_call value_and_jacobian!(f!, y_in, jac_in, ba, x, extras)
    @test_opt value_and_jacobian!(f!, y_in, jac_in, ba, x, extras)
end

end
