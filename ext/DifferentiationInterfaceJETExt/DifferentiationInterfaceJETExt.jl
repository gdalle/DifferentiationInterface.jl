module DifferentiationInterfaceJETExt

using ADTypes: AbstractADType
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest: Scenario
using JET: @test_call, @test_opt
using LinearAlgebra: LinearAlgebra
using Test: @testset, @test

## Pushforward

function test_jet(ba::AbstractADType, ::typeof(pushforward), scen::Scenario{false};)
    (; f, x, dx, dy) = deepcopy(scen)
    extras = prepare_pushforward(f, ba, x)
    dy_in = zero(dy)

    @test_opt value_and_pushforward!!(f, dy_in, ba, x, dx, extras)
    @test_opt value_and_pushforward(f, ba, x, dx, extras)
    return nothing
end

function test_jet(ba::AbstractADType, ::typeof(pushforward), scen::Scenario{true};)
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(f!, ba, y, x)
    y_in = zero(y)
    dy_in = zero(dy)

    @test_opt value_and_pushforward!!(f!, y_in, dy_in, ba, x, dx, extras)
    return nothing
end

## Pullback

function test_jet(ba::AbstractADType, ::typeof(pullback), scen::Scenario{false};)
    (; f, x, dx, dy) = deepcopy(scen)
    extras = prepare_pullback(f, ba, x)
    dx_in = zero(dx)

    @test_opt value_and_pullback!!(f, dx_in, ba, x, dy, extras)
    @test_opt value_and_pullback(f, ba, x, dy, extras)
    return nothing
end

function test_jet(ba::AbstractADType, ::typeof(pullback), scen::Scenario{true};)
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pullback(f!, ba, y, x)
    y_in = zero(y)
    dx_in = zero(dx)

    @test_opt value_and_pullback!!(f!, y_in, dx_in, ba, x, dy, extras)
    return nothing
end

## Derivative

function test_jet(ba::AbstractADType, ::typeof(derivative), scen::Scenario{false};)
    (; f, x, dy) = deepcopy(scen)
    extras = prepare_derivative(f, ba, x)
    der_in = zero(dy)

    @test_opt value_and_derivative!!(f, der_in, ba, x, extras)
    @test_opt value_and_derivative(f, ba, x, extras)
    return nothing
end

function test_jet(ba::AbstractADType, ::typeof(derivative), scen::Scenario{true};)
    (; f, x, y, dy) = deepcopy(scen)
    f! = f
    extras = prepare_derivative(f!, ba, y, x)
    y_in = zero(y)
    der_in = zero(dy)

    @test_opt value_and_derivative!!(f!, y_in, der_in, ba, x, extras)
    return nothing
end

## Gradient

function test_jet(ba::AbstractADType, ::typeof(gradient), scen::Scenario{false};)
    (; f, x, dx) = deepcopy(scen)
    extras = prepare_gradient(f, ba, x)
    grad_in = zero(dx)

    @test_opt value_and_gradient!!(f, grad_in, ba, x, extras)
    @test_opt value_and_gradient(f, ba, x, extras)
    return nothing
end

## Jacobian

function test_jet(ba::AbstractADType, ::typeof(jacobian), scen::Scenario{false};)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)
    jac_in = similar(x, length(y), length(x))

    @test_opt value_and_jacobian!!(f, jac_in, ba, x, extras)
    @test_opt value_and_jacobian(f, ba, x, extras)
    return nothing
end

function test_jet(ba::AbstractADType, ::typeof(jacobian), scen::Scenario{true};)
    (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, ba, y, x)
    y_in = zero(y)
    jac_in = similar(x, length(y), length(x))

    @test_opt value_and_jacobian!!(f!, y_in, jac_in, ba, x, extras)
    return nothing
end

## Second derivative

function test_jet(ba::AbstractADType, ::typeof(second_derivative), scen::Scenario{false};)
    (; f, x) = deepcopy(scen)
    extras = prepare_second_derivative(f, ba, x)

    @test_opt second_derivative(f, ba, x, extras)
    return nothing
end

## HVP

function test_jet(ba::AbstractADType, ::typeof(hvp), scen::Scenario{false};)
    (; f, x, dx) = deepcopy(scen)
    extras = prepare_hvp(f, ba, x)

    @test_opt hvp(f, ba, x, dx, extras)
    return nothing
end

## Hessian

function test_jet(ba::AbstractADType, ::typeof(hessian), scen::Scenario{false};)
    (; f, x) = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)

    @test_opt hessian(f, ba, x, extras)
    return nothing
end

end #module
