module DifferentiationInterfaceJETExt

using ADTypes: AbstractADType
using DifferentiationInterface
using DifferentiationInterface:
    mode, supports_mutation, supports_pushforward, supports_pullback
using DifferentiationInterface.DifferentiationTest: Scenario
import DifferentiationInterface.DifferentiationTest as DT
using JET: @test_call, @test_opt
using LinearAlgebra: LinearAlgebra
using Test: @testset, @test

## Pushforward

function DT.test_type_stability(
    ba::AbstractADType, ::typeof(value_and_pushforward), scen::Scenario{false}
)
    (; f, x, dx, dy) = deepcopy(scen)
    extras = prepare_pushforward(f, ba, x)
    dy_in = zero(dy)

    @test_call value_and_pushforward!(f, dy_in, ba, x, dx, extras)
    @test_opt value_and_pushforward!(f, dy_in, ba, x, dx, extras)

    @test_call value_and_pushforward(f, ba, x, dx, extras)
    @test_opt value_and_pushforward(f, ba, x, dx, extras)
end

function DT.test_type_stability(
    ba::AbstractADType, ::typeof(value_and_pushforward), scen::Scenario{true}
)
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(f!, ba, y, x)
    y_in = zero(y)
    dy_in = zero(dy)

    @test_call value_and_pushforward!(f!, y_in, dy_in, ba, x, dx, extras)
    @test_opt value_and_pushforward!(f!, y_in, dy_in, ba, x, dx, extras)
end

## Pullback

function DT.test_type_stability(
    ba::AbstractADType, ::typeof(value_and_pullback), scen::Scenario{false}
)
    (; f, x, dx, dy) = deepcopy(scen)
    extras = prepare_pullback(f, ba, x)
    dx_in = zero(dx)

    @test_call value_and_pullback!(f, dx_in, ba, x, dy, extras)
    @test_opt value_and_pullback!(f, dx_in, ba, x, dy, extras)

    @test_call value_and_pullback(f, ba, x, dy, extras)
    @test_opt value_and_pullback(f, ba, x, dy, extras)
end

function DT.test_type_stability(
    ba::AbstractADType, ::typeof(value_and_pullback), scen::Scenario{true}
)
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pullback(f!, ba, y, x)
    y_in = zero(y)
    dx_in = zero(dx)

    @test_call value_and_pullback!(f!, y_in, dx_in, ba, x, dy, extras)
    @test_opt value_and_pullback!(f!, y_in, dx_in, ba, x, dy, extras)
end

## Derivative

function DT.test_type_stability(
    ba::AbstractADType, ::typeof(value_and_derivative), scen::Scenario{false}
)
    (; f, x, dy) = deepcopy(scen)
    extras = prepare_derivative(f, ba, x)
    der_in = zero(dy)

    @test_call value_and_derivative!(f, der_in, ba, x, extras)
    @test_opt value_and_derivative!(f, der_in, ba, x, extras)

    @test_call value_and_derivative(f, ba, x, extras)
    @test_opt value_and_derivative(f, ba, x, extras)
end

function DT.test_type_stability(
    ba::AbstractADType, ::typeof(value_and_derivative), scen::Scenario{true}
)
    (; f, x, y, dy) = deepcopy(scen)
    f! = f
    extras = prepare_derivative(f!, ba, y, x)
    y_in = zero(y)
    der_in = zero(dy)

    @test_call value_and_derivative!(f!, y_in, der_in, ba, x, extras)
    @test_opt value_and_derivative!(f!, y_in, der_in, ba, x, extras)
end

## Gradient

function DT.test_type_stability(
    ba::AbstractADType, ::typeof(value_and_gradient), scen::Scenario{false}
)
    (; f, x, dx) = deepcopy(scen)
    extras = prepare_gradient(f, ba, x)
    grad_in = zero(dx)

    @test_call value_and_gradient!(f, grad_in, ba, x, extras)
    @test_opt value_and_gradient!(f, grad_in, ba, x, extras)

    @test_call value_and_gradient(f, ba, x, extras)
    @test_opt value_and_gradient(f, ba, x, extras)
end

## Jacobian

function DT.test_type_stability(
    ba::AbstractADType, ::typeof(value_and_jacobian), scen::Scenario{false}
)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)
    jac_in = zeros(eltype(y), length(y), length(x))

    @test_call value_and_jacobian!(f, jac_in, ba, x, extras)
    @test_opt value_and_jacobian!(f, jac_in, ba, x, extras)

    @test_call value_and_jacobian(f, ba, x, extras)
    @test_opt value_and_jacobian(f, ba, x, extras)
end

function DT.test_type_stability(
    ba::AbstractADType, ::typeof(value_and_jacobian), scen::Scenario{true}
)
    (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, ba, y, x)
    y_in = zero(y)
    jac_in = zeros(eltype(y), length(y), length(x))

    @test_call value_and_jacobian!(f!, y_in, jac_in, ba, x, extras)
    @test_opt value_and_jacobian!(f!, y_in, jac_in, ba, x, extras)
end

end #module
