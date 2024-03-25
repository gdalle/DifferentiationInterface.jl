module DifferentiationInterfaceCorrectnessTestExt

using ADTypes: AbstractADType
using DifferentiationInterface
using DifferentiationInterface: myisapprox, mysimilar
using DifferentiationInterface.DifferentiationTest: Scenario
import DifferentiationInterface.DifferentiationTest as DT
using ForwardDiff: ForwardDiff
using LinearAlgebra: dot
using Test: @testset, @test
using Zygote: Zygote

function test_scen_intact(new_scen, scen)
    @testset "Scenario intact" begin
        @test myisapprox(new_scen.x, scen.x)
        @test myisapprox(new_scen.y, scen.y)
        @test myisapprox(new_scen.dx, scen.dx)
        @test myisapprox(new_scen.dy, scen.dy)
    end
end

## Pushforward

function test_correctness(ba::AbstractADType, ::typeof(pushforward), scen::Scenario{false})
    (; f, x, y, dx, dy) = new_scen = deepcopy(scen)
    dy_true = true_pushforward(f, x, y, dx; mutating=false)

    y_out1, dy_out1 = value_and_pushforward(f, ba, x, dx)
    dy_in2 = mysimilar(dy)
    y_out2, dy_out2 = value_and_pushforward!!(f, dy_in2, ba, x, dx)

    dy_out3 = pushforward(f, ba, x, dx)
    dy_in4 = mysimilar(dy)
    dy_out4 = pushforward!!(f, dy_in4, ba, x, dx)

    @testset "Primal value" begin
        @test myisapprox(y_out1, y)
        @test myisapprox(y_out2, y)
    end
    @testset "Tangent value" begin
        @test myisapprox(dy_out1, dy_true; rtol=1e-3)
        @test myisapprox(dy_out2, dy_true; rtol=1e-3)
        @test myisapprox(dy_out3, dy_true; rtol=1e-3)
        @test myisapprox(dy_out4, dy_true; rtol=1e-3)
    end
    return test_scen_intact(new_scen, scen)
end

function test_correctness(ba::AbstractADType, ::typeof(pushforward), scen::Scenario{true})
    (; f, x, y, dx, dy) = new_scen = deepcopy(scen)
    f! = f
    dy_true = true_pushforward(f!, x, y, dx; mutating=true)

    y_in = mysimilar(y)
    dy_in = mysimilar(dy)
    y_out, dy_out = value_and_pushforward!!(f!, y_in, dy_in, ba, x, dx)

    @testset "Primal value" begin
        @test myisapprox(y_out, y)
        @testset "Mutation" begin
            if ismutable(y)
                @test myisapprox(y_in, y)
            end
        end
    end
    @testset "Tangent value" begin
        @test myisapprox(dy_out, dy_true; rtol=1e-3)
    end
    return test_scen_intact(new_scen, scen)
end

## Pullback

function test_correctness(ba::AbstractADType, ::typeof(pullback), scen::Scenario{false})
    (; f, x, y, dx, dy) = new_scen = deepcopy(scen)
    dx_true = true_pullback(f, x, y, dy; mutating=false)

    y_out1, dx_out1 = value_and_pullback(f, ba, x, dy)
    dx_in2 = mysimilar(dx)
    y_out2, dx_out2 = value_and_pullback!!(f, dx_in2, ba, x, dy)

    dx_out3 = pullback(f, ba, x, dy)
    dx_in4 = mysimilar(dx)
    dx_out4 = pullback!!(f, dx_in4, ba, x, dy)

    @testset "Primal value" begin
        @test myisapprox(y_out1, y)
        @test myisapprox(y_out2, y)
    end
    @testset "Cotangent value" begin
        @test myisapprox(dx_out1, dx_true; rtol=1e-3)
        @test myisapprox(dx_out2, dx_true; rtol=1e-3)
        @test myisapprox(dx_out3, dx_true; rtol=1e-3)
        @test myisapprox(dx_out4, dx_true; rtol=1e-3)
    end
    return test_scen_intact(new_scen, scen)
end

function test_correctness(ba::AbstractADType, ::typeof(pullback), scen::Scenario{true})
    (; f, x, y, dx, dy) = new_scen = deepcopy(scen)
    f! = f
    dx_true = true_pullback(f, x, y, dy; mutating=true)

    y_in = mysimilar(y)
    dx_in = mysimilar(dx)
    y_out, dx_out = value_and_pullback!!(f!, y_in, dx_in, ba, x, dy)

    @testset "Primal value" begin
        @test myisapprox(y_out, y)
        @testset "Mutation" begin
            if ismutable(y)
                @test myisapprox(y_in, y)
            end
        end
    end
    @testset "Cotangent value" begin
        @test myisapprox(dx_out, dx_true; rtol=1e-3)
    end
    return test_scen_intact(new_scen, scen)
end

## Derivative

function test_correctness(ba::AbstractADType, ::typeof(derivative), scen::Scenario{false})
    (; f, x, y, dx, dy) = new_scen = deepcopy(scen)
    der_true = ForwardDiff.derivative(f, x)

    y_out1, der_out1 = value_and_derivative(f, ba, x)
    der_in2 = mysimilar(dy)
    y_out2, der_out2 = value_and_derivative!!(f, der_in2, ba, x)

    der_out3 = derivative(f, ba, x)
    der_in4 = mysimilar(dy)
    der_out4 = derivative!!(f, der_in4, ba, x)

    @testset "Primal value" begin
        @test myisapprox(y_out1, y)
        @test myisapprox(y_out2, y)
    end
    @testset "Derivative value" begin
        @test myisapprox(der_out1, der_true; rtol=1e-3)
        @test myisapprox(der_out2, der_true; rtol=1e-3)
        @test myisapprox(der_out3, der_true; rtol=1e-3)
        @test myisapprox(der_out4, der_true; rtol=1e-3)
    end
    return test_scen_intact(new_scen, scen)
end

function test_correctness(ba::AbstractADType, ::typeof(derivative), scen::Scenario{true})
    (; f, x, y, dx, dy) = new_scen = deepcopy(scen)
    f! = f
    der_true = ForwardDiff.derivative(f!, y, x)

    y_in = mysimilar(y)
    der_in = mysimilar(dy)
    y_out, der_out = value_and_derivative!!(f!, y_in, der_in, ba, x)

    @testset "Primal value" begin
        @test myisapprox(y_out, y)
        @testset "Mutation" begin
            if ismutable(y)
                @test myisapprox(y_in, y)
            end
        end
    end
    @testset "Derivative value" begin
        @test myisapprox(der_out, der_true; rtol=1e-3)
    end
    return test_scen_intact(new_scen, scen)
end

## Gradient

function test_correctness(ba::AbstractADType, ::typeof(gradient), scen::Scenario{false})
    (; f, x, y, dx, dy) = new_scen = deepcopy(scen)
    grad_true = if x isa Number
        ForwardDiff.derivative(f, x)
    else
        only(Zygote.gradient(f, x))
    end

    y_out1, grad_out1 = value_and_gradient(f, ba, x)
    grad_in2 = mysimilar(dx)
    y_out2, grad_out2 = value_and_gradient!!(f, grad_in2, ba, x)

    grad_out3 = gradient(f, ba, x)
    grad_in4 = mysimilar(dx)
    grad_out4 = gradient!!(f, grad_in4, ba, x)

    @testset "Primal value" begin
        @test myisapprox(y_out1, y)
        @test myisapprox(y_out2, y)
    end
    @testset "Gradient value" begin
        @test myisapprox(grad_out1, grad_true; rtol=1e-3)
        @test myisapprox(grad_out2, grad_true; rtol=1e-3)
        @test myisapprox(grad_out3, grad_true; rtol=1e-3)
        @test myisapprox(grad_out4, grad_true; rtol=1e-3)
    end
    return test_scen_intact(new_scen, scen)
end

## Jacobian

function test_correctness(ba::AbstractADType, ::typeof(jacobian), scen::Scenario{false})
    (; f, x, y) = new_scen = deepcopy(scen)
    jac_true = ForwardDiff.jacobian(f, x)

    y_out1, jac_out1 = value_and_jacobian(f, ba, x)
    jac_in2 = mysimilar(jac_true)
    y_out2, jac_out2 = value_and_jacobian!!(f, jac_in2, ba, x)

    jac_out3 = jacobian(f, ba, x)
    jac_in4 = mysimilar(jac_true)
    jac_out4 = jacobian!!(f, jac_in4, ba, x)

    @testset "Primal value" begin
        @test myisapprox(y_out1, y)
        @test myisapprox(y_out2, y)
    end
    @testset "Jacobian value" begin
        @test myisapprox(jac_out1, jac_true; rtol=1e-3)
        @test myisapprox(jac_out2, jac_true; rtol=1e-3)
        @test myisapprox(jac_out3, jac_true; rtol=1e-3)
        @test myisapprox(jac_out4, jac_true; rtol=1e-3)
        @testset "Mutation" begin
            @test myisapprox(jac_in2, jac_true; rtol=1e-3)
            @test myisapprox(jac_in4, jac_true; rtol=1e-3)
        end
    end
    return test_scen_intact(new_scen, scen)
end

function test_correctness(ba::AbstractADType, ::typeof(jacobian), scen::Scenario{true})
    (; f, x, y) = new_scen = deepcopy(scen)
    f! = f
    jac_true = ForwardDiff.jacobian(f!, y, x)

    y_in = mysimilar(y)
    jac_in = mysimilar(jac_true)
    y_out, jac_out = value_and_jacobian!!(f!, y_in, jac_in, ba, x)

    @testset "Primal value" begin
        @test myisapprox(y_out, y)
        @testset "Mutation" begin
            @test myisapprox(y_in, y)
        end
    end
    @testset "Jacobian value" begin
        @test myisapprox(jac_out, jac_true; rtol=1e-3)
        @testset "Mutation" begin
            @test myisapprox(jac_in, jac_true; rtol=1e-3)
        end
    end
    return test_scen_intact(new_scen, scen)
end

## Second derivative

function test_correctness(ba::AbstractADType, ::typeof(second_derivative), scen::Scenario)
    (; f, x) = deepcopy(scen)
    der2_true = ForwardDiff.derivative(z -> ForwardDiff.derivative(f, z), x)

    der2_out1 = second_derivative(f, ba, x)

    @testset "Second derivative value" begin
        @test myisapprox(der2_out1, der2_true; rtol=1e-3)
    end
end

## Hessian-vector product

function test_correctness(ba::AbstractADType, ::typeof(hvp), scen::Scenario)
    (; f, x, dx) = deepcopy(scen)
    hess_true = if x isa Number
        ForwardDiff.derivative(z -> ForwardDiff.derivative(f, z), x)
    else
        ForwardDiff.hessian(f, x)
    end
    hvp_true = if x isa Number
        hess_true * dx
    else
        reshape((hess_true * vec(dx)), size(x))
    end

    hvp_out1 = hvp(f, ba, x, dx)

    @testset "Hessian-vector product value" begin
        @test myisapprox(hvp_out1, hvp_true; rtol=1e-3)
    end
end

## Hessian

function test_correctness(ba::AbstractADType, ::typeof(hessian), scen::Scenario)
    (; f, x, y) = deepcopy(scen)
    hess_true = ForwardDiff.hessian(f, x)

    hess_out1 = hessian(f, ba, x)

    @testset "Hessian value" begin
        @test myisapprox(hess_out1, hess_true; rtol=1e-3)
    end
end

## Utils

function true_pushforward(f, x::Number, y::Number, dx; mutating)
    return ForwardDiff.derivative(f, x) * dx
end

function true_pushforward(f, x::Number, y::AbstractArray, dx; mutating)
    if mutating
        return ForwardDiff.derivative(f, deepcopy(y), x) .* dx
    else
        return ForwardDiff.derivative(f, x) .* dx
    end
end

function true_pushforward(f, x::AbstractArray, y::Number, dx; mutating)
    return dot(Zygote.gradient(f, x)[1], dx)
end

function true_pushforward(f, x::AbstractArray, y::AbstractArray, dx; mutating)
    if mutating
        return reshape(ForwardDiff.jacobian(f, deepcopy(y), x) * vec(dx), size(y))
    else
        return reshape(ForwardDiff.jacobian(f, x) * vec(dx), size(y))
    end
end

function true_pullback(f, x::Number, y::Number, dy; mutating)
    return ForwardDiff.derivative(f, x) * dy
end

function true_pullback(f, x::Number, y::AbstractArray, dy; mutating)
    if mutating
        return dot(ForwardDiff.derivative(f, deepcopy(y), x), dy)
    else
        return dot(ForwardDiff.derivative(f, x), dy)
    end
end

function true_pullback(f, x::AbstractArray, y::Number, dy; mutating)
    return Zygote.gradient(f, x)[1] .* dy
end

function true_pullback(f, x::AbstractArray, y::AbstractArray, dy; mutating)
    if mutating
        return reshape(
            transpose(ForwardDiff.jacobian(f, deepcopy(y), x)) * vec(dy), size(x)
        )
    else
        return reshape(transpose(ForwardDiff.jacobian(f, x)) * vec(dy), size(x))
    end
end

end # module
