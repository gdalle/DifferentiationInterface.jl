module DifferentiationInterfaceJETExt

using ADTypes:
    AbstractADType,
    AbstractFiniteDifferencesMode,
    AbstractForwardMode,
    AbstractReverseMode,
    AbstractSymbolicDifferentiationMode
using DifferentiationInterface
using DifferentiationInterface:
    inner,
    mode,
    outer,
    supports_mutation,
    supports_pushforward,
    supports_pullback,
    supports_hvp
using DifferentiationInterface.DifferentiationTest
import DifferentiationInterface.DifferentiationTest as DT
using DifferentiationInterface.DifferentiationTest:
    AbstractOperator,
    PushforwardAllocating,
    PushforwardMutating,
    PullbackAllocating,
    PullbackMutating,
    MultiderivativeAllocating,
    MultiderivativeMutating,
    GradientAllocating,
    JacobianAllocating,
    JacobianMutating,
    DerivativeAllocating,
    SecondDerivativeAllocating,
    HessianAllocating,
    HessianVectorProductAllocating,
    filter_compatible
using JET: @test_call, @test_opt
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
                @testset "$s" for s in filter_compatible(op, scenarios)
                    test_type(op, backend, s)
                end
            end
        end
    end
end

## Pushforward

function test_type(::PushforwardAllocating, ba::AbstractADType, scen::Scenario)
    Bool(supports_pushforward(ba)) || return nothing
    (; f, x, dx, dy) = deepcopy(scen)
    dy_in = zero(dy)

    @test_call value_and_pushforward!(dy_in, ba, f, x, dx)
    @test_opt value_and_pushforward!(dy_in, ba, f, x, dx)

    @test_call pushforward!(dy_in, ba, f, x, dx)
    @test_opt pushforward!(dy_in, ba, f, x, dx)

    @test_call value_and_pushforward(ba, f, x, dx)
    @test_opt value_and_pushforward(ba, f, x, dx)

    @test_call pushforward(ba, f, x, dx)
    @test_opt pushforward(ba, f, x, dx)
end

function test_type(::PushforwardMutating, ba::AbstractADType, scen::Scenario)
    Bool(supports_pushforward(ba)) || return nothing
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    y_in = zero(y)
    dy_in = zero(dy)

    @test_call value_and_pushforward!(y_in, dy_in, ba, f!, x, dx)
    @test_opt value_and_pushforward!(y_in, dy_in, ba, f!, x, dx)
end

## Pullback

function test_type(::PullbackAllocating, ba::AbstractADType, scen::Scenario)
    Bool(supports_pullback(ba)) || return nothing
    (; f, x, dx, dy) = deepcopy(scen)
    dx_in = zero(dx)

    @test_call value_and_pullback!(dx_in, ba, f, x, dy)
    @test_opt value_and_pullback!(dx_in, ba, f, x, dy)

    @test_call pullback!(dx_in, ba, f, x, dy)
    @test_opt pullback!(dx_in, ba, f, x, dy)

    @test_call value_and_pullback(ba, f, x, dy)
    @test_opt value_and_pullback(ba, f, x, dy)

    @test_call pullback(ba, f, x, dy)
    @test_opt pullback(ba, f, x, dy)
end

function test_type(::PullbackMutating, ba::AbstractADType, scen::Scenario)
    Bool(supports_pullback(ba)) || return nothing
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    y_in = zero(y)
    dx_in = zero(dx)

    @test_call value_and_pullback!(y_in, dx_in, ba, f!, x, dy)
    @test_opt value_and_pullback!(y_in, dx_in, ba, f!, x, dy)
end

## Derivative

function test_type(::DerivativeAllocating, ba::AbstractADType, scen::Scenario)
    (; f, x) = deepcopy(scen)

    @test_call value_and_derivative(ba, f, x)
    @test_opt value_and_derivative(ba, f, x)

    @test_call derivative(ba, f, x)
    @test_opt derivative(ba, f, x)
end

## Multiderivative

function test_type(::MultiderivativeAllocating, ba::AbstractADType, scen::Scenario)
    (; f, x, dy) = deepcopy(scen)
    multider_in = zero(dy)

    @test_call value_and_multiderivative!(multider_in, ba, f, x)
    @test_opt value_and_multiderivative!(multider_in, ba, f, x)

    @test_call multiderivative!(multider_in, ba, f, x)
    @test_opt multiderivative!(multider_in, ba, f, x)

    @test_call value_and_multiderivative(ba, f, x)
    @test_opt value_and_multiderivative(ba, f, x)

    @test_call multiderivative(ba, f, x)
    @test_opt multiderivative(ba, f, x)
end

function test_type(::MultiderivativeMutating, ba::AbstractADType, scen::Scenario)
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dy) = deepcopy(scen)
    f! = f
    y_in = zero(y)
    multider_in = zero(dy)

    @test_call value_and_multiderivative!(y_in, multider_in, ba, f!, x)
    @test_opt value_and_multiderivative!(y_in, multider_in, ba, f!, x)
end

## Gradient

function test_type(::GradientAllocating, ba::AbstractADType, scen::Scenario)
    (; f, x, dx) = deepcopy(scen)
    grad_in = zero(dx)

    @test_call value_and_gradient!(grad_in, ba, f, x)
    @test_opt value_and_gradient!(grad_in, ba, f, x)

    @test_call gradient!(grad_in, ba, f, x)
    @test_opt gradient!(grad_in, ba, f, x)

    @test_call value_and_gradient(ba, f, x)
    @test_opt value_and_gradient(ba, f, x)

    @test_call gradient(ba, f, x)
    @test_opt gradient(ba, f, x)
end

## Jacobian

function test_type(::JacobianAllocating, ba::AbstractADType, scen::Scenario)
    (; f, x, y) = deepcopy(scen)
    jac_in = zeros(eltype(y), length(y), length(x))

    @test_call value_and_jacobian!(jac_in, ba, f, x)
    @test_opt value_and_jacobian!(jac_in, ba, f, x)

    @test_call jacobian!(jac_in, ba, f, x)
    @test_opt jacobian!(jac_in, ba, f, x)

    @test_call value_and_jacobian(ba, f, x)
    @test_opt value_and_jacobian(ba, f, x)

    @test_call jacobian(ba, f, x)
    @test_opt jacobian(ba, f, x)
end

function test_type(::JacobianMutating, ba::AbstractADType, scen::Scenario)
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y) = deepcopy(scen)
    f! = f
    y_in = zero(y)
    jac_in = zeros(eltype(y), length(y), length(x))

    @test_call value_and_jacobian!(y_in, jac_in, ba, f!, x)
    @test_opt value_and_jacobian!(y_in, jac_in, ba, f!, x)
end

## Second derivative

function test_type(::SecondDerivativeAllocating, ba::AbstractADType, scen::Scenario)
    (; f, x) = deepcopy(scen)

    @test_call value_derivative_and_second_derivative(ba, f, x)
    @test_opt value_derivative_and_second_derivative(ba, f, x)

    @test_call second_derivative(ba, f, x)
    @test_opt second_derivative(ba, f, x)
end

## Hessian-vector product

function test_type(::HessianVectorProductAllocating, ba::AbstractADType, scen::Scenario)
    Bool(supports_hvp(ba)) || return nothing
    (; f, x, dx) = deepcopy(scen)
    grad_in = zero(dx)
    hvp_in = zero(dx)

    @test_call ignored_modules = (LinearAlgebra,) hessian_vector_product!(
        hvp_in, ba, f, x, dx
    )
    @test_opt ignored_modules = (LinearAlgebra,) hessian_vector_product!(
        hvp_in, ba, f, x, dx
    )

    # @test_call ignored_modules = (LinearAlgebra,) hessian_vector_product(ba, f, x, dx)
    # @test_opt ignored_modules = (LinearAlgebra,) hessian_vector_product(ba, f, x, dx)

    # @test_call ignored_modules = (LinearAlgebra,) gradient_and_hessian_vector_product!(
    #     grad_in, hvp_in, ba, f, x, dx
    # )
    # @test_opt ignored_modules = (LinearAlgebra,) gradient_and_hessian_vector_product!(
    #     grad_in, hvp_in, ba, f, x, dx
    # )

    # @test_call ignored_modules = (LinearAlgebra,) gradient_and_hessian_vector_product(
    #     ba, f, x, dx
    # )
    # @test_opt ignored_modules = (LinearAlgebra,) gradient_and_hessian_vector_product(
    #     ba, f, x, dx
    # )
end

## Hessian

function test_type(::HessianAllocating, ba::AbstractADType, scen::Scenario)
    (; f, x, dx) = deepcopy(scen)
    grad_in = zero(dx)
    hess_in = zeros(eltype(x), length(x), length(x))

    @test_call ignored_modules = (LinearAlgebra,) value_gradient_and_hessian!(
        grad_in, hess_in, ba, f, x
    )
    @test_opt ignored_modules = (LinearAlgebra,) value_gradient_and_hessian!(
        grad_in, hess_in, ba, f, x
    )

    @test_call ignored_modules = (LinearAlgebra,) hessian!(hess_in, ba, f, x)
    @test_opt ignored_modules = (LinearAlgebra,) hessian!(hess_in, ba, f, x)

    @test_call ignored_modules = (LinearAlgebra,) value_gradient_and_hessian(ba, f, x)
    @test_opt ignored_modules = (LinearAlgebra,) value_gradient_and_hessian(ba, f, x)

    @test_call ignored_modules = (LinearAlgebra,) hessian(ba, f, x)
    @test_opt ignored_modules = (LinearAlgebra,) hessian(ba, f, x)
end

end
