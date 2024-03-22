module DifferentiationInterfaceForwardDiffTestExt

using ADTypes: AbstractADType
using DifferentiationInterface
using DifferentiationInterface:
    mode, myeltype, supports_mutation, supports_pushforward, supports_pullback
using DifferentiationInterface.DifferentiationTest:
    Scenario, allocating, backend_string, mutating, scalar_in, scalar_out, array_array
import DifferentiationInterface.DifferentiationTest as DT
using ForwardDiff: ForwardDiff
using LinearAlgebra: dot
using Test: @testset, @test

function DT.test_correctness(
    backends::Vector{<:AbstractADType},
    operators::Vector{Symbol},
    scenarios::Vector{<:Scenario};
)
    @testset verbose = true "Correctness" begin
        @testset verbose = true "$(backend_string(backend))" for backend in backends
            @testset "$op" for op in operators
                if op == :pushforward_allocating
                    @testset "$s" for s in allocating(scenarios)
                        test_correctness_pushforward_allocating(backend, s)
                    end
                elseif op == :pushforward_mutating
                    @testset "$s" for s in mutating(scenarios)
                        test_correctness_pushforward_mutating(backend, s)
                    end

                elseif op == :pullback_allocating
                    @testset "$s" for s in allocating(scenarios)
                        test_correctness_pullback_allocating(backend, s)
                    end
                elseif op == :pullback_mutating
                    @testset "$s" for s in mutating(scenarios)
                        test_correctness_pullback_mutating(backend, s)
                    end

                elseif op == :derivative_allocating
                    @testset "$s" for s in allocating(scalar_in(scenarios))
                        test_correctness_derivative_allocating(backend, s)
                    end
                elseif op == :derivative_mutating
                    @testset "$s" for s in mutating(scalar_in(scenarios))
                        test_correctness_derivative_mutating(backend, s)
                    end

                elseif op == :gradient_allocating
                    @testset "$s" for s in allocating(scalar_out(scenarios))
                        test_correctness_gradient_allocating(backend, s)
                    end

                elseif op == :jacobian_allocating
                    @testset "$s" for s in allocating(array_array(scenarios))
                        test_correctness_jacobian_allocating(backend, s)
                    end
                elseif op == :jacobian_mutating
                    @testset "$s" for s in mutating(array_array(scenarios))
                        test_correctness_jacobian_mutating(backend, s)
                    end

                else
                    throw(ArgumentError("Invalid operator to test: `:$op`"))
                end
            end
        end
    end
end

function test_scen_intact(new_scen, scen)
    @testset "Scenario intact" begin
        @test new_scen.x ≈ scen.x
        @test new_scen.y ≈ scen.y
        @test new_scen.dx ≈ scen.dx
        @test new_scen.dy ≈ scen.dy
    end
end

## Pushforward

function test_correctness_pushforward_allocating(ba::AbstractADType, scen::Scenario)
    Bool(supports_pushforward(ba)) || return nothing
    (; f, x, y, dx) = new_scen = deepcopy(scen)
    dy_true = true_pushforward(f, x, y, dx; mutating=false)

    y_out1, dy_out1 = value_and_pushforward(f, ba, x, dx)
    dy_in2 = zero(dy_out1)
    y_out2, dy_out2 = value_and_pushforward!(f, dy_in2, ba, x, dx)

    @testset "Primal value" begin
        @test y_out1 ≈ y
        @test y_out2 ≈ y
    end
    @testset "Tangent value" begin
        @test dy_out1 ≈ dy_true rtol = 1e-3
        @test dy_out2 ≈ dy_true rtol = 1e-3
        if ismutable(dy_true)
            @testset "Mutation" begin
                @test dy_in2 ≈ dy_true rtol = 1e-3
            end
        end
    end
    return test_scen_intact(new_scen, scen)
end

function test_correctness_pushforward_mutating(ba::AbstractADType, scen::Scenario)
    Bool(supports_pushforward(ba)) || return nothing
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dx) = new_scen = deepcopy(scen)
    f! = f
    dy_true = true_pushforward(f!, x, y, dx; mutating=true)

    y_in = zero(y)
    dy_in = zero(dy_true)
    y_out, dy_out = value_and_pushforward!(f!, y_in, dy_in, ba, x, dx)

    @testset "Primal value" begin
        @test y_out ≈ y
        @testset "Mutation" begin
            if ismutable(y)
                @test y_in ≈ y
            end
        end
    end
    @testset "Tangent value" begin
        @test dy_out ≈ dy_true rtol = 1e-3
        @testset "Mutation" begin
            if ismutable(dy_true)
                @test dy_in ≈ dy_true rtol = 1e-3
            end
        end
    end
    return test_scen_intact(new_scen, scen)
end

## Pullback

function test_correctness_pullback_allocating(ba::AbstractADType, scen::Scenario)
    Bool(supports_pullback(ba)) || return nothing
    (; f, x, y, dy) = new_scen = deepcopy(scen)
    dx_true = true_pullback(f, x, y, dy; mutating=false)

    y_out1, dx_out1 = value_and_pullback(f, ba, x, dy)
    dx_in2 = zero(dx_out1)
    y_out2, dx_out2 = value_and_pullback!(f, dx_in2, ba, x, dy)

    @testset "Primal value" begin
        @test y_out1 ≈ y
        @test y_out2 ≈ y
    end
    @testset "Cotangent value" begin
        @test dx_out1 ≈ dx_true rtol = 1e-3
        @test dx_out2 ≈ dx_true rtol = 1e-3
        if ismutable(dx_true)
            @testset "Mutation" begin
                @test dx_in2 ≈ dx_true rtol = 1e-3
            end
        end
    end
    return test_scen_intact(new_scen, scen)
end

function test_correctness_pullback_mutating(ba::AbstractADType, scen::Scenario)
    Bool(supports_pullback(ba)) || return nothing
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y, dy) = new_scen = deepcopy(scen)
    f! = f
    dx_true = true_pullback(f, x, y, dy; mutating=true)

    y_in = zero(y)
    dx_in = zero(dx_true)
    y_out, dx_out = value_and_pullback!(f!, y_in, dx_in, ba, x, dy)

    @testset "Primal value" begin
        @test y_out ≈ y
        @testset "Mutation" begin
            if ismutable(y)
                @test y_in ≈ y
            end
        end
    end
    @testset "Cotangent value" begin
        @test dx_out ≈ dx_true rtol = 1e-3
        if ismutable(dx_true)
            @testset "Mutation" begin
                if ismutable(dx_true)
                    @test dx_in ≈ dx_true rtol = 1e-3
                end
            end
        end
    end
    return test_scen_intact(new_scen, scen)
end

## Derivative

function test_correctness_derivative_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, y) = new_scen = deepcopy(scen)
    der_true = ForwardDiff.derivative(f, x)

    y_out1, der_out1 = value_and_derivative(f, ba, x)
    der_in2 = zero(der_out1)
    y_out2, der_out2 = value_and_derivative!(f, der_in2, ba, x)

    @testset "Primal value" begin
        @test y_out1 ≈ y
        @test y_out2 ≈ y
    end
    @testset "Derivative value" begin
        @test der_out1 ≈ der_true rtol = 1e-3
        @test der_out2 ≈ der_true rtol = 1e-3
        @testset "Mutation" begin
            if ismutable(der_true)
                @test der_in2 ≈ der_true rtol = 1e-3
            end
        end
    end
    return test_scen_intact(new_scen, scen)
end

function test_correctness_derivative_mutating(ba::AbstractADType, scen::Scenario)
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y) = new_scen = deepcopy(scen)
    f! = f
    der_true = ForwardDiff.derivative(f!, y, x)

    y_in = zero(y)
    der_in = zero(der_true)
    y_out, der_out = value_and_derivative!(f!, y_in, der_in, ba, x)

    @testset "Primal value" begin
        @test y_out ≈ y
        @testset "Mutation" begin
            if ismutable(y)
                @test y_in ≈ y
            end
        end
    end
    @testset "Derivative value" begin
        @test der_out ≈ der_true rtol = 1e-3
        @testset "Mutation" begin
            if ismutable(der_true)
                @test der_in ≈ der_true rtol = 1e-3
            end
        end
    end
    return test_scen_intact(new_scen, scen)
end

## Gradient

function test_correctness_gradient_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, y) = new_scen = deepcopy(scen)
    grad_true = if x isa Number
        ForwardDiff.derivative(f, x)
    elseif x isa AbstractArray
        ForwardDiff.gradient(f, x)
    end

    y_out1, grad_out1 = value_and_gradient(f, ba, x)
    grad_in2 = zero(grad_out1)
    y_out2, grad_out2 = value_and_gradient!(f, grad_in2, ba, x)

    @testset "Primal value" begin
        @test y_out1 ≈ y
        @test y_out2 ≈ y
    end
    @testset "Gradient value" begin
        @test grad_out1 ≈ grad_true rtol = 1e-3
        @test grad_out2 ≈ grad_true rtol = 1e-3
        @testset "Mutation" begin
            if ismutable(grad_true)
                @test grad_in2 ≈ grad_true rtol = 1e-3
            end
        end
    end
    return test_scen_intact(new_scen, scen)
end

## Jacobian

function test_correctness_jacobian_allocating(ba::AbstractADType, scen::Scenario)
    (; f, x, y) = new_scen = deepcopy(scen)
    jac_true = ForwardDiff.jacobian(f, x)

    y_out1, jac_out1 = value_and_jacobian(f, ba, x)
    jac_in2 = zero(jac_out1)
    y_out2, jac_out2 = value_and_jacobian!(f, jac_in2, ba, x)

    @testset "Primal value" begin
        @test y_out1 ≈ y
        @test y_out2 ≈ y
    end
    @testset "Jacobian value" begin
        @test jac_out1 ≈ jac_true rtol = 1e-3
        @test jac_out2 ≈ jac_true rtol = 1e-3
        @testset "Mutation" begin
            @test jac_in2 ≈ jac_true rtol = 1e-3
        end
    end
    return test_scen_intact(new_scen, scen)
end

function test_correctness_jacobian_mutating(ba::AbstractADType, scen::Scenario)
    Bool(supports_mutation(ba)) || return nothing
    (; f, x, y) = new_scen = deepcopy(scen)
    f! = f
    jac_true = ForwardDiff.jacobian(f!, y, x)

    y_in = zero(y)
    jac_in = similar(y, length(y), length(x))
    y_out, jac_out = value_and_jacobian!(f!, y_in, jac_in, ba, x)

    @testset "Primal value" begin
        @test y_out ≈ y
        @testset "Mutation" begin
            @test y_in ≈ y
        end
    end
    @testset "Jacobian value" begin
        @test jac_out ≈ jac_true rtol = 1e-3
        @testset "Mutation" begin
            @test jac_in ≈ jac_true rtol = 1e-3
        end
    end
    return test_scen_intact(new_scen, scen)
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
    return dot(ForwardDiff.gradient(f, x), dx)
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
    return ForwardDiff.gradient(f, x) .* dy
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
