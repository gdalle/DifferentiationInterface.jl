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
    dy_in = zero(dy)

    @test_call value_and_pushforward!(f, dy_in, ba, x, dx)
    @test_opt value_and_pushforward!(f, dy_in, ba, x, dx)

    @test_call value_and_pushforward(f, ba, x, dx)
    @test_opt value_and_pushforward(f, ba, x, dx)
end

function test_type_pushforward_mutating(ba::AbstractADType, scen::Scenario)
    Bool(supports_pushforward(ba)) || return nothing
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    y_in = zero(y)
    dy_in = zero(dy)

    @test_call value_and_pushforward!(f!, y_in, dy_in, ba, x, dx)
    @test_opt value_and_pushforward!(f!, y_in, dy_in, ba, x, dx)
end

## Pullback

function test_type_pullback_allocating(ba::AbstractADType, scen::Scenario)
    Bool(supports_pullback(ba)) || return nothing
    (; f, x, dx, dy) = deepcopy(scen)
    dx_in = zero(dx)

    @test_call value_and_pullback!(f, dx_in, ba, x, dy)
    @test_opt value_and_pullback!(f, dx_in, ba, x, dy)

    @test_call value_and_pullback(f, ba, x, dy)
    @test_opt value_and_pullback(f, ba, x, dy)
end

function test_type_pullback_mutating(ba::AbstractADType, scen::Scenario)
    Bool(supports_pullback(ba)) || return nothing
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    y_in = zero(y)
    dx_in = zero(dx)

    @test_call value_and_pullback!(f!, y_in, dx_in, ba, x, dy)
    @test_opt value_and_pullback!(f!, y_in, dx_in, ba, x, dy)
end

## Derivative

function test_type_derivative_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, dy) = deepcopy(scen)
    der_in = zero(dy)

    @test_call value_and_derivative!(f, der_in, ba, x)
    @test_opt value_and_derivative!(f, der_in, ba, x)

    @test_call value_and_derivative(f, ba, x)
    @test_opt value_and_derivative(f, ba, x)
end

function test_type_derivative_mutating(ba::AbstractADType, scen::Scenario)
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dy) = deepcopy(scen)
    f! = f
    y_in = zero(y)
    der_in = zero(dy)

    @test_call value_and_derivative!(f!, y_in, der_in, ba, x)
    @test_opt value_and_derivative!(f!, y_in, der_in, ba, x)
end

## Gradient

function test_type_gradient_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, dx) = deepcopy(scen)
    grad_in = zero(dx)

    @test_call value_and_gradient!(f, grad_in, ba, x)
    @test_opt value_and_gradient!(f, grad_in, ba, x)

    @test_call value_and_gradient(f, ba, x)
    @test_opt value_and_gradient(f, ba, x)
end

## Jacobian

function test_type_jacobian_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, y) = deepcopy(scen)
    jac_in = zeros(eltype(y), length(y), length(x))

    @test_call value_and_jacobian!(f, jac_in, ba, x)
    @test_opt value_and_jacobian!(f, jac_in, ba, x)

    @test_call value_and_jacobian(f, ba, x)
    @test_opt value_and_jacobian(f, ba, x)
end

function test_type_jacobian_mutating(ba::AbstractADType, scen::Scenario)
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y) = deepcopy(scen)
    f! = f
    y_in = zero(y)
    jac_in = zeros(eltype(y), length(y), length(x))

    @test_call value_and_jacobian!(f!, y_in, jac_in, ba, x)
    @test_opt value_and_jacobian!(f!, y_in, jac_in, ba, x)
end

end
