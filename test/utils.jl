using DifferentiationInterface
using DifferentiationInterface:
    AbstractBackend, AbstractReverseBackend, AbstractForwardBackend
using ForwardDiff: ForwardDiff
using LinearAlgebra
using JET
using Random: AbstractRNG, randn!
using StableRNGs
using Test

## Test scenarios

@kwdef struct Scenario{F,X,Y,D1,D2,D3,D4}
    "function"
    f::F
    "argument"
    x::X
    "primal value"
    y::Y
    "pushforward seed"
    dx::X
    "pullback seed"
    dy::Y
    "pullback result"
    dx_true::X
    "pushforward result"
    dy_true::Y
    "derivative result"
    der_true::D1 = nothing
    "multiderivative result"
    multider_true::D2 = nothing
    "gradient result"
    grad_true::D3 = nothing
    "Jacobian result"
    jac_true::D4 = nothing
end

## Constructors

function make_scenario(rng::AbstractRNG, f, x)
    y = f(x)
    return make_scenario(rng, f, x, y)
end

function make_scenario(rng::AbstractRNG, f::F, x::X, y::Y) where {F,X<:Number,Y<:Number}
    dx = randn(rng, X)
    dy = randn(rng, Y)
    der_true = ForwardDiff.derivative(f, x)
    dx_true = der_true * dy
    dy_true = der_true * dx
    return Scenario(; f, x, y, dx, dy, dx_true, dy_true, der_true)
end

function make_scenario(
    rng::AbstractRNG, f::F, x::X, y::Y
) where {F,X<:Number,Y<:AbstractArray}
    dx = randn(rng, X)
    dy = similar(y)
    randn!(rng, dy)
    multider_true = ForwardDiff.derivative(f, x)
    dx_true = dot(multider_true, dy)
    dy_true = multider_true .* dx
    return Scenario(; f, x, y, dx, dy, dx_true, dy_true, multider_true)
end

function make_scenario(
    rng::AbstractRNG, f::F, x::X, y::Y
) where {F,X<:AbstractArray,Y<:Number}
    dx = similar(x)
    randn!(rng, dx)
    dy = randn(rng, Y)
    grad_true = ForwardDiff.gradient(f, x)
    dx_true = grad_true .* dy
    dy_true = dot(grad_true, dx)
    return Scenario(; f, x, y, dx, dy, dx_true, dy_true, grad_true)
end

function make_scenario(
    rng::AbstractRNG, f::F, x::X, y::Y
) where {F,X<:AbstractArray,Y<:AbstractArray}
    dx = similar(x)
    randn!(rng, dx)
    dy = similar(y)
    randn!(rng, dy)
    jac_true = ForwardDiff.jacobian(f, x)
    dx_true = reshape(transpose(jac_true) * vec(dy), size(x))
    dy_true = reshape(jac_true * vec(dx), size(y))
    return Scenario(; f, x, y, dx, dy, dx_true, dy_true, jac_true)
end

## Access

get_input_type(::Scenario{F,X}) where {F,X} = X
get_output_type(::Scenario{F,X,Y}) where {F,X,Y} = Y

## Seed

rng = StableRNG(63)

## Scenarios

f_scalar_scalar(x::Number)::Number = sin(x)
f_scalar_vector(x::Number)::AbstractVector = [sin(x), sin(2x)]
f_scalar_matrix(x::Number)::AbstractMatrix = hcat([sin(x) cos(x)], [sin(2x) cos(2x)])

function f_vector_scalar(x::AbstractVector)::Number
    a = eachindex(x)
    return sum(sin.(a .* x))
end

function f_matrix_scalar(x::AbstractMatrix)::Number
    a, b = axes(x)
    return sum(sin.(a .* x)) + sum(cos.(transpose(b) .* x))
end

function f_vector_vector(x::AbstractVector)::AbstractVector
    a = eachindex(x)
    return vcat(sin.(a .* x), cos.(a .* x))
end

function f_vector_matrix(x::AbstractVector)::AbstractMatrix
    a = eachindex(x)
    return hcat(sin.(a .* x), cos.(a .* x))
end

function f_matrix_vector(x::AbstractMatrix)::AbstractVector
    a, b = axes(x)
    return vcat(vec(sin.(a .* x)), vec(cos.(transpose(b) .* x)))
end

function f_matrix_matrix(x::AbstractMatrix)::AbstractMatrix
    a, b = axes(x)
    return hcat(vec(sin.(a .* x)), vec(cos.(transpose(b) .* x)))
end

## All

scenarios = [
    make_scenario(rng, f_scalar_scalar, 1.0),
    make_scenario(rng, f_scalar_vector, 1.0),
    make_scenario(rng, f_scalar_matrix, 1.0),
    make_scenario(rng, f_vector_scalar, [1.0, 2.0]),
    make_scenario(rng, f_matrix_scalar, [1.0 2.0; 3.0 4.0]),
    make_scenario(rng, f_vector_vector, [1.0, 2.0]),
    make_scenario(rng, f_vector_matrix, [1.0, 2.0]),
    make_scenario(rng, f_matrix_vector, [1.0 2.0; 3.0 4.0]),
    make_scenario(rng, f_matrix_matrix, [1.0 2.0; 3.0 4.0]),
];

## Test utilities

function test_pushforward(
    backend::AbstractForwardBackend,
    scenarios::Vector{<:Scenario};
    input_type::Type=Any,
    output_type::Type=Any,
    allocs::Bool=false,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        get_input_type(s) <: input_type && get_output_type(s) <: output_type
    end
    @testset "Pushforward ($(is_custom(backend) ? "custom" : "fallback"))" begin
        for scenario in scenarios
            X, Y = get_input_type(scenario), get_output_type(scenario)
            handles_types(backend, X, Y) || continue

            @testset "$X -> $Y" begin
                (; f, x, y, dx, dy_true) = scenario
                y_out1, dy_out1 = value_and_pushforward(backend, f, x, dx)
                dy_in2 = zero(dy_out1)
                y_out2, dy_out2 = value_and_pushforward!(dy_in2, backend, f, x, dx)

                dy_out3 = pushforward(backend, f, x, dx)
                dy_in4 = zero(dy_out3)
                dy_out4 = pushforward!(dy_in4, backend, f, x, dx)

                @testset "Primal value" begin
                    @test y_out1 ≈ y
                    @test y_out2 ≈ y
                end
                @testset "Tangent value" begin
                    @test dy_out1 ≈ dy_true rtol = 1e-3
                    @test dy_out2 ≈ dy_true rtol = 1e-3
                    @test dy_out3 ≈ dy_true rtol = 1e-3
                    @test dy_out4 ≈ dy_true rtol = 1e-3
                    if ismutable(dy_true)
                        @testset "Mutation" begin
                            @test dy_in2 ≈ dy_true rtol = 1e-3
                            @test dy_in4 ≈ dy_true rtol = 1e-3
                        end
                    end
                end
                allocs && @testset "Allocations" begin
                    @test iszero(@allocated value_and_pushforward!(dy_in2, backend, f, x, dx))
                    @test iszero(@allocated pushforward!(dy_in4, backend, f, x, dx))
                end
                type_stability && @testset "Type stability" begin
                    @test_opt value_and_pushforward!(dy_in2, backend, f, x, dx)
                    @test_opt pushforward!(dy_in4, backend, f, x, dx)
                end
            end
        end
    end
end

function test_pullback(
    backend::AbstractReverseBackend,
    scenarios::Vector{<:Scenario};
    input_type::Type=Any,
    output_type::Type=Any,
    allocs::Bool=false,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        (get_input_type(s) <: input_type) && (get_output_type(s) <: output_type)
    end
    @testset "Pullback ($(is_custom(backend) ? "custom" : "fallback"))" begin
        for scenario in scenarios
            X, Y = get_input_type(scenario), get_output_type(scenario)
            handles_types(backend, X, Y) || continue

            @testset "$X -> $Y" begin
                (; f, x, y, dy, dx_true) = scenario

                y_out1, dx_out1 = value_and_pullback(backend, f, x, dy)
                dx_in2 = zero(dx_out1)
                y_out2, dx_out2 = value_and_pullback!(dx_in2, backend, f, x, dy)

                dx_out3 = pullback(backend, f, x, dy)
                dx_in4 = zero(dx_out3)
                dx_out4 = pullback!(dx_in4, backend, f, x, dy)

                @testset "Primal value" begin
                    @test y_out1 ≈ y
                    @test y_out2 ≈ y
                end
                @testset "Cotangent value" begin
                    @test dx_out1 ≈ dx_true rtol = 1e-3
                    @test dx_out2 ≈ dx_true rtol = 1e-3
                    @test dx_out3 ≈ dx_true rtol = 1e-3
                    @test dx_out4 ≈ dx_true rtol = 1e-3
                    if ismutable(dx_true)
                        @testset "Mutation" begin
                            @test dx_in2 ≈ dx_true rtol = 1e-3
                            @test dx_in4 ≈ dx_true rtol = 1e-3
                        end
                    end
                end
                allocs && @testset "Allocations" begin
                    @test iszero(@allocated value_and_pullback!(dx_in2, backend, f, x, dy))
                    @test iszero(@allocated pullback!(dx_in4, backend, f, x, dy))
                end
                type_stability && @testset "Type stability" begin
                    @test_opt value_and_pullback!(dx_in2, backend, f, x, dy)
                    @test_opt pullback!(dx_in4, backend, f, x, dy)
                end
            end
        end
    end
end

function test_derivative(
    backend::AbstractBackend,
    scenarios::Vector{<:Scenario};
    allocs::Bool=false,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        (get_input_type(s) <: Number) && (get_output_type(s) <: Number)
    end
    @testset "Derivative ($(is_custom(backend) ? "custom" : "fallback"))" begin
        for scenario in scenarios
            X, Y = get_input_type(scenario), get_output_type(scenario)
            handles_types(backend, X, Y) || continue

            @testset "$X -> $Y" begin
                (; f, x, y, der_true) = scenario

                y_out1, der_out1 = value_and_derivative(backend, f, x)

                der_out2 = derivative(backend, f, x)

                @testset "Primal value" begin
                    @test y_out1 ≈ y
                end
                @testset "Derivative value" begin
                    @test der_out1 ≈ der_true rtol = 1e-3
                    @test der_out2 ≈ der_true rtol = 1e-3
                end
                allocs && @testset "Allocations" begin
                    @test iszero(@allocated value_and_derivative(backend, f, x))
                    @test iszero(@allocated derivative(backend, f, x))
                end
                type_stability && @testset "Type stability" begin
                    @test_opt value_and_derivative(backend, f, x)
                    @test_opt derivative(backend, f, x)
                end
            end
        end
    end
end

function test_multiderivative(
    backend::AbstractBackend,
    scenarios::Vector{<:Scenario};
    allocs::Bool=false,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        (get_input_type(s) <: Number) && (get_output_type(s) <: AbstractArray)
    end
    @testset "Multiderivative ($(is_custom(backend) ? "custom" : "fallback"))" begin
        for scenario in scenarios
            X, Y = get_input_type(scenario), get_output_type(scenario)
            handles_types(backend, X, Y) || continue

            @testset "$X -> $Y" begin
                (; f, x, y, multider_true) = scenario

                y_out1, multider_out1 = value_and_multiderivative(backend, f, x)
                multider_in2 = zero(multider_out1)
                y_out2, multider_out2 = value_and_multiderivative!(
                    multider_in2, backend, f, x
                )

                multider_out3 = multiderivative(backend, f, x)
                multider_in4 = zero(multider_out3)
                multider_out4 = multiderivative!(multider_in4, backend, f, x)

                @testset "Primal value" begin
                    @test y_out1 ≈ y
                    @test y_out2 ≈ y
                end
                @testset "Multiderivative value" begin
                    @test multider_out1 ≈ multider_true rtol = 1e-3
                    @test multider_out2 ≈ multider_true rtol = 1e-3
                    @test multider_out3 ≈ multider_true rtol = 1e-3
                    @test multider_out4 ≈ multider_true rtol = 1e-3
                    @testset "Mutation" begin
                        @test multider_in2 ≈ multider_true rtol = 1e-3
                        @test multider_in4 ≈ multider_true rtol = 1e-3
                    end
                end
                allocs && @testset "Allocations" begin
                    @test iszero(
                        @allocated value_and_multiderivative!(multider_in2, backend, f, x)
                    )
                    @test iszero(@allocated multiderivative!(multider_in4, backend, f, x))
                end
                type_stability && @testset "Type stability" begin
                    @test_opt value_and_multiderivative!(multider_in2, backend, f, x)
                    @test_opt multiderivative!(multider_in4, backend, f, x)
                end
            end
        end
    end
end

function test_gradient(
    backend::AbstractBackend,
    scenarios::Vector{<:Scenario};
    allocs::Bool=false,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        (get_input_type(s) <: AbstractArray) && (get_output_type(s) <: Number)
    end
    @testset "Gradient ($(is_custom(backend) ? "custom" : "fallback"))" begin
        for scenario in scenarios
            X, Y = get_input_type(scenario), get_output_type(scenario)
            handles_types(backend, X, Y) || continue

            @testset "$X -> $Y" begin
                (; f, x, y, grad_true) = scenario

                y_out1, grad_out1 = value_and_gradient(backend, f, x)
                grad_in2 = zero(grad_out1)
                y_out2, grad_out2 = value_and_gradient!(grad_in2, backend, f, x)

                grad_out3 = gradient(backend, f, x)
                grad_in4 = zero(grad_out3)
                grad_out4 = gradient!(grad_in4, backend, f, x)

                @testset "Primal value" begin
                    @test y_out1 ≈ y
                    @test y_out2 ≈ y
                end
                @testset "Gradient value" begin
                    @test grad_out1 ≈ grad_true rtol = 1e-3
                    @test grad_out2 ≈ grad_true rtol = 1e-3
                    @test grad_out3 ≈ grad_true rtol = 1e-3
                    @test grad_out4 ≈ grad_true rtol = 1e-3
                    @testset "Mutation" begin
                        @test grad_in2 ≈ grad_true rtol = 1e-3
                        @test grad_in4 ≈ grad_true rtol = 1e-3
                    end
                end
                allocs && @testset "Allocations" begin
                    @test iszero(@allocated value_and_gradient!(grad_in2, backend, f, x))
                    @test iszero(@allocated gradient!(grad_in4, backend, f, x))
                end
                type_stability && @testset "Type stability" begin
                    @test_opt value_and_gradient!(grad_in2, backend, f, x)
                    @test_opt gradient!(grad_in4, backend, f, x)
                end
            end
        end
    end
end

function test_jacobian(
    backend::AbstractBackend,
    scenarios::Vector{<:Scenario};
    allocs::Bool=false,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        (get_input_type(s) <: AbstractArray) && (get_output_type(s) <: AbstractArray)
    end
    @testset "Jacobian ($(is_custom(backend) ? "custom" : "fallback"))" begin
        for scenario in scenarios
            X, Y = get_input_type(scenario), get_output_type(scenario)
            handles_types(backend, X, Y) || continue

            @testset "$X -> $Y" begin
                (; f, x, y, jac_true) = scenario

                y_out1, jac_out1 = value_and_jacobian(backend, f, x)
                jac_in2 = zero(jac_out1)
                y_out2, jac_out2 = value_and_jacobian!(jac_in2, backend, f, x)

                jac_out3 = jacobian(backend, f, x)
                jac_in4 = zero(jac_out3)
                jac_out4 = jacobian!(jac_in4, backend, f, x)

                @testset "Primal value" begin
                    @test y_out1 ≈ y
                    @test y_out2 ≈ y
                end
                @testset "Jacobian value" begin
                    @test jac_out1 ≈ jac_true rtol = 1e-3
                    @test jac_out2 ≈ jac_true rtol = 1e-3
                    @test jac_out3 ≈ jac_true rtol = 1e-3
                    @test jac_out4 ≈ jac_true rtol = 1e-3
                    @testset "Mutation" begin
                        @test jac_in2 ≈ jac_true rtol = 1e-3
                        @test jac_in4 ≈ jac_true rtol = 1e-3
                    end
                end
                allocs && @testset "Allocations" begin
                    @test iszero(@allocated value_and_jacobian!(jac_in2, backend, f, x))
                    @test iszero(@allocated jacobian!(jac_in4, backend, f, x))
                end
                type_stability && @testset "Type stability" begin
                    @test_opt value_and_jacobian!(jac_in2, backend, f, x)
                    @test_opt jacobian!(jac_in4, backend, f, x)
                end
            end
        end
    end
end

function test_jacobian_and_friends(
    backend::AbstractBackend,
    scenarios::Vector{<:Scenario};
    input_type::Type=Any,
    output_type::Type=Any,
    allocs::Bool=false,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        (get_input_type(s) <: input_type) && (get_output_type(s) <: output_type)
    end
    test_derivative(backend, scenarios; allocs, type_stability)
    test_multiderivative(backend, scenarios; allocs, type_stability)
    test_gradient(backend, scenarios; allocs, type_stability)
    test_jacobian(backend, scenarios; allocs, type_stability)
    return nothing
end
