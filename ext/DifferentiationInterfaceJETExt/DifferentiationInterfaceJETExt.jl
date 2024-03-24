module DifferentiationInterfaceJETExt

using ADTypes: AbstractADType
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest: Scenario
import DifferentiationInterface.DifferentiationTest as DT
using JET: @test_call, @test_opt
using LinearAlgebra: LinearAlgebra
using Test: @testset, @test

## Pushforward

function DT.test_jet(
    ba::AbstractADType, ::typeof(pushforward), scen::Scenario{false}; call::Bool, opt::Bool
)
    (; f, x, dx, dy) = deepcopy(scen)
    extras = prepare_pushforward(f, ba, x)
    dy_in = zero(dy)

    call && value_and_pushforward!!(f, dy_in, ba, x, dx, extras)
    opt && @test_opt value_and_pushforward!!(f, dy_in, ba, x, dx, extras)

    call && value_and_pushforward(f, ba, x, dx, extras)
    opt && @test_opt value_and_pushforward(f, ba, x, dx, extras)
    return nothing
end

function DT.test_jet(
    ba::AbstractADType, ::typeof(pushforward), scen::Scenario{true}; call::Bool, opt::Bool
)
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pushforward(f!, ba, y, x)
    y_in = zero(y)
    dy_in = zero(dy)

    call && value_and_pushforward!!(f!, y_in, dy_in, ba, x, dx, extras)
    opt && @test_opt value_and_pushforward!!(f!, y_in, dy_in, ba, x, dx, extras)
    return nothing
end

## Pullback

function DT.test_jet(
    ba::AbstractADType, ::typeof(pullback), scen::Scenario{false}; call::Bool, opt::Bool
)
    (; f, x, dx, dy) = deepcopy(scen)
    extras = prepare_pullback(f, ba, x)
    dx_in = zero(dx)

    call && value_and_pullback!!(f, dx_in, ba, x, dy, extras)
    opt && @test_opt value_and_pullback!!(f, dx_in, ba, x, dy, extras)

    call && value_and_pullback(f, ba, x, dy, extras)
    opt && @test_opt value_and_pullback(f, ba, x, dy, extras)
    return nothing
end

function DT.test_jet(
    ba::AbstractADType, ::typeof(pullback), scen::Scenario{true}; call::Bool, opt::Bool
)
    (; f, x, y, dx, dy) = deepcopy(scen)
    f! = f
    extras = prepare_pullback(f!, ba, y, x)
    y_in = zero(y)
    dx_in = zero(dx)

    call && value_and_pullback!!(f!, y_in, dx_in, ba, x, dy, extras)
    opt && @test_opt value_and_pullback!!(f!, y_in, dx_in, ba, x, dy, extras)
    return nothing
end

## Derivative

function DT.test_jet(
    ba::AbstractADType, ::typeof(derivative), scen::Scenario{false}; call::Bool, opt::Bool
)
    (; f, x, dy) = deepcopy(scen)
    extras = prepare_derivative(f, ba, x)
    der_in = zero(dy)

    call && value_and_derivative!!(f, der_in, ba, x, extras)
    opt && @test_opt value_and_derivative!!(f, der_in, ba, x, extras)

    call && value_and_derivative(f, ba, x, extras)
    opt && @test_opt value_and_derivative(f, ba, x, extras)
    return nothing
end

function DT.test_jet(
    ba::AbstractADType, ::typeof(derivative), scen::Scenario{true}; call::Bool, opt::Bool
)
    (; f, x, y, dy) = deepcopy(scen)
    f! = f
    extras = prepare_derivative(f!, ba, y, x)
    y_in = zero(y)
    der_in = zero(dy)

    call && value_and_derivative!!(f!, y_in, der_in, ba, x, extras)
    opt && @test_opt value_and_derivative!!(f!, y_in, der_in, ba, x, extras)
    return nothing
end

## Gradient

function DT.test_jet(
    ba::AbstractADType, ::typeof(gradient), scen::Scenario{false}; call::Bool, opt::Bool
)
    (; f, x, dx) = deepcopy(scen)
    extras = prepare_gradient(f, ba, x)
    grad_in = zero(dx)

    call && value_and_gradient!!(f, grad_in, ba, x, extras)
    opt && @test_opt value_and_gradient!!(f, grad_in, ba, x, extras)

    call && value_and_gradient(f, ba, x, extras)
    opt && @test_opt value_and_gradient(f, ba, x, extras)
    return nothing
end

## Jacobian

function DT.test_jet(
    ba::AbstractADType, ::typeof(jacobian), scen::Scenario{false}; call::Bool, opt::Bool
)
    (; f, x, y) = deepcopy(scen)
    extras = prepare_jacobian(f, ba, x)
    jac_in = similar(x, length(y), length(x))

    call && value_and_jacobian!!(f, jac_in, ba, x, extras)
    opt && @test_opt value_and_jacobian!!(f, jac_in, ba, x, extras)

    call && value_and_jacobian!!(f, jac_in, ba, x, extras)
    opt && @test_opt value_and_jacobian(f, ba, x, extras)
    return nothing
end

function DT.test_jet(
    ba::AbstractADType, ::typeof(jacobian), scen::Scenario{true}; call::Bool, opt::Bool
)
    (; f, x, y) = deepcopy(scen)
    f! = f
    extras = prepare_jacobian(f!, ba, y, x)
    y_in = zero(y)
    jac_in = similar(x, length(y), length(x))

    call && value_and_jacobian!!(f!, y_in, jac_in, ba, x, extras)
    opt && @test_opt value_and_jacobian!!(f!, y_in, jac_in, ba, x, extras)
    return nothing
end

## Second derivative

function DT.test_jet(
    ba::AbstractADType,
    ::typeof(second_derivative),
    scen::Scenario{false};
    call::Bool,
    opt::Bool,
)
    (; f, x) = deepcopy(scen)
    extras = prepare_second_derivative(f, ba, x)

    call && second_derivative(f, ba, x, extras)
    opt && @test_opt second_derivative(f, ba, x, extras)
    return nothing
end

## HVP

function DT.test_jet(
    ba::AbstractADType, ::typeof(hvp), scen::Scenario{false}; call::Bool, opt::Bool
)
    (; f, x, dx) = deepcopy(scen)
    extras = prepare_hvp(f, ba, x)

    call && hvp(f, ba, x, dx, extras)
    opt && @test_opt hvp(f, ba, x, dx, extras)
    return nothing
end

## Hessian

function DT.test_jet(
    ba::AbstractADType, ::typeof(hessian), scen::Scenario{false}; call::Bool, opt::Bool
)
    (; f, x) = deepcopy(scen)
    extras = prepare_hessian(f, ba, x)

    call && hessian(f, ba, x, extras)
    opt && @test_opt hessian(f, ba, x, extras)
    return nothing
end

end #module
