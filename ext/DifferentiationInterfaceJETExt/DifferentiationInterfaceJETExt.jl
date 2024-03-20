module DifferentiationInterfaceJETExt

using ADTypes: AbstractADType
using DifferentiationInterface
using DifferentiationInterface:
    ForwardMode,
    ReverseMode,
    MutationSupported,
    MutationNotSupported,
    mode,
    mutation_behavior,
    outer
using DifferentiationInterface.DifferentiationTest
import DifferentiationInterface.DifferentiationTest as DT
using JET: @test_opt
using LinearAlgebra: LinearAlgebra
using Test

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
                    @testset "$s" for s in allocating(scalar_scalar(scenarios))
                        test_type_derivative_allocating(backend, s)
                    end

                elseif op == :multiderivative_allocating
                    @testset "$s" for s in allocating(scalar_array(scenarios))
                        test_type_multiderivative_allocating(backend, s)
                    end
                elseif op == :multiderivative_mutating
                    @testset "$s" for s in mutating(scalar_array(scenarios))
                        test_type_multiderivative_mutating(backend, s)
                    end

                elseif op == :gradient_allocating
                    @testset "$s" for s in allocating(array_scalar(scenarios))
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

                elseif op == :second_derivative_allocating
                    @testset "$s" for s in allocating(scalar_scalar(scenarios))
                        test_type_second_derivative_allocating(backend, s)
                    end

                elseif op == :hessian_vector_product_allocating
                    @testset "$s" for s in allocating(array_scalar(scenarios))
                        test_type_hessian_vector_product_allocating(backend, s)
                    end
                elseif op == :hessian_allocating
                    @testset "$s" for s in allocating(array_scalar(scenarios))
                        test_type_hessian_allocating(backend, s)
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
    isa(mode(ba), ReverseMode) && return nothing
    (; f, x, dx, dy) = deepcopy(scen)
    dy_in = zero(dy)
    @test_opt value_and_pushforward!(dy_in, ba, f, x, dx)
    @test_opt pushforward!(dy_in, ba, f, x, dx)
    @test_opt value_and_pushforward(ba, f, x, dx)
    @test_opt pushforward(ba, f, x, dx)
end

function test_type_pushforward_mutating(ba::AbstractADType, scen::Scenario)
    isa(mode(ba), ReverseMode) && return nothing
    isa(mutation_behavior(ba), MutationNotSupported) && return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    y_in = zero(y)
    dy_in = zero(dy)
    @test_opt value_and_pushforward!(y_in, dy_in, ba, f!, x, dx)
end

## Pullback

function test_type_pullback_allocating(ba::AbstractADType, scen::Scenario)
    isa(mode(ba), ForwardMode) && return nothing
    (; f, x, dx, dy) = deepcopy(scen)
    dx_in = zero(dx)
    @test_opt value_and_pullback!(dx_in, ba, f, x, dy)
    @test_opt pullback!(dx_in, ba, f, x, dy)
    @test_opt value_and_pullback(ba, f, x, dy)
    @test_opt pullback(ba, f, x, dy)
end

function test_type_pullback_mutating(ba::AbstractADType, scen::Scenario)
    isa(mode(ba), ForwardMode) && return nothing
    isa(mutation_behavior(ba), MutationNotSupported) && return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    y_in = zero(y)
    dx_in = zero(dx)
    @test_opt value_and_pullback!(y_in, dx_in, ba, f!, x, dy)
end

## Derivative

function test_type_derivative_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x) = deepcopy(scen)
    @test_opt value_and_derivative(ba, f, x)
    @test_opt derivative(ba, f, x)
end

## Multiderivative

function test_type_multiderivative_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, dy) = deepcopy(scen)
    multider_in = zero(dy)
    @test_opt value_and_multiderivative!(multider_in, ba, f, x)
    @test_opt multiderivative!(multider_in, ba, f, x)
    @test_opt value_and_multiderivative(ba, f, x)
    @test_opt multiderivative(ba, f, x)
end

function test_type_multiderivative_mutating(ba::AbstractADType, scen::Scenario)
    isa(mutation_behavior(ba), MutationNotSupported) && return nothing
    (; f, x, y, dy) = deepcopy(scen)
    f! = f
    y_in = zero(y)
    multider_in = zero(dy)
    @test_opt value_and_multiderivative!(y_in, multider_in, ba, f!, x)
end

## Gradient

function test_type_gradient_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, dx) = deepcopy(scen)
    grad_in = zero(dx)
    @test_opt value_and_gradient!(grad_in, ba, f, x)
    @test_opt gradient!(grad_in, ba, f, x)
    @test_opt value_and_gradient(ba, f, x)
    @test_opt gradient(ba, f, x)
end

## Jacobian

function test_type_jacobian_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, y) = deepcopy(scen)
    jac_in = zeros(eltype(y), length(y), length(x))
    @test_opt value_and_jacobian!(jac_in, ba, f, x)
    @test_opt jacobian!(jac_in, ba, f, x)
    @test_opt value_and_jacobian(ba, f, x)
    @test_opt jacobian(ba, f, x)
end

function test_type_jacobian_mutating(ba::AbstractADType, scen::Scenario)
    isa(mutation_behavior(ba), MutationNotSupported) && return nothing
    (; f, x, y) = deepcopy(scen)
    f! = f
    y_in = zero(y)
    jac_in = zeros(eltype(y), length(y), length(x))
    @test_opt value_and_jacobian!(y_in, jac_in, ba, f!, x)
end

## Second derivative

function test_type_second_derivative_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x) = deepcopy(scen)
    @test_opt value_derivative_and_second_derivative(ba, f, x)
    @test_opt second_derivative(ba, f, x)
end

## Hessian-vector product

function test_type_hessian_vector_product_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, dx) = deepcopy(scen)
    grad_in = zero(dx)
    hvp_in = zero(dx)
    @test_opt ignored_modules = (LinearAlgebra,) hessian_vector_product!(
        hvp_in, ba, f, x, dx
    )
    @test_opt ignored_modules = (LinearAlgebra,) hessian_vector_product(ba, f, x, dx)
    @test_opt ignored_modules = (LinearAlgebra,) gradient_and_hessian_vector_product!(
        grad_in, hvp_in, ba, f, x, dx
    )
    @test_opt ignored_modules = (LinearAlgebra,) gradient_and_hessian_vector_product(
        ba, f, x, dx
    )
end

## Hessian

function test_type_hessian_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, dx) = deepcopy(scen)
    grad_in = zero(dx)
    hess_in = zeros(eltype(x), length(x), length(x))
    @test_opt ignored_modules = (LinearAlgebra,) value_gradient_and_hessian!(
        grad_in, hess_in, ba, f, x
    )
    @test_opt ignored_modules = (LinearAlgebra,) hessian!(hess_in, ba, f, x)
    @test_opt ignored_modules = (LinearAlgebra,) value_gradient_and_hessian(ba, f, x)
    @test_opt ignored_modules = (LinearAlgebra,) hessian(ba, f, x)
end

end
