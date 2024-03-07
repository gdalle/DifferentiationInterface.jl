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
    "primal output"
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
    scenarios::Vector{<:Scenario}=scenarios;
    input_type::Type=Any,
    output_type::Type=Any,
    allocs::Bool=false,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        get_input_type(s) <: input_type && get_output_type(s) <: output_type
    end
    @testset "Pushforward" begin
        for scenario in scenarios
            X, Y = get_input_type(scenario), get_output_type(scenario)
            @testset "$X -> $Y" begin
                (; f, x, y, dx, dy_true) = scenario
                y_out, dy_out = value_and_pushforward(backend, f, x, dx)

                @testset "Primal output" begin
                    @test y_out ≈ y
                end
                @testset "Tangent output" begin
                    @test dy_out ≈ dy_true rtol = 1e-3
                end
                if ismutable(dy_out)
                    @testset "Mutation" begin
                        dy_in = similar(dy_out)
                        value_and_pushforward!(dy_in, backend, f, x, dx)
                        @test dy_in ≈ dy_true rtol = 1e-3
                    end
                end
                if allocs
                    @testset "Allocations" begin
                        @test (@allocated value_and_pushforward!(
                            dy_out, backend, f, x, dx
                        )) == 0
                    end
                end
                if type_stability
                    @testset "Type stability" begin
                        @test_opt value_and_pushforward!(dy_out, backend, f, x, dx)
                    end
                end
            end
        end
    end
end

function test_pullback(
    backend::AbstractReverseBackend,
    scenarios=scenarios;
    input_type::Type=Any,
    output_type::Type=Any,
    allocs::Bool=false,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        (get_input_type(s) <: input_type) && (get_output_type(s) <: output_type)
    end
    @testset "Pullback" begin
        for scenario in scenarios
            X, Y = get_input_type(scenario), get_output_type(scenario)
            @testset "$X -> $Y" begin
                (; f, x, y, dy, dx_true) = scenario
                y_out, dx_out = value_and_pullback(backend, f, x, dy)

                @testset "Primal output" begin
                    @test y_out ≈ y
                end
                @testset "Co-tangent output" begin
                    @test dx_out ≈ dx_true rtol = 1e-3
                end
                if ismutable(dx_out)
                    @testset "Mutation" begin
                        dx_in = similar(dx_out)
                        value_and_pullback!(dx_in, backend, f, x, dy)
                        @test dx_in ≈ dx_true rtol = 1e-3
                    end
                end
                if allocs
                    @testset "Allocations" begin
                        @test (@allocated value_and_pullback!(dx_out, backend, f, x, dy)) ==
                            0
                    end
                end
                if type_stability
                    @testset "Type stability" begin
                        @test_opt value_and_pullback!(dx_out, backend, f, x, dy)
                    end
                end
            end
        end
    end
end

function test_derivative(
    backend::AbstractBackend{custom},
    scenarios::Vector{<:Scenario}=scenarios;
    allocs::Bool=false,
    type_stability::Bool=true,
) where {custom}
    scenarios = filter(scenarios) do s
        (get_input_type(s) <: Number) && (get_output_type(s) <: Number)
    end
    @testset "Derivative ($(custom ? "custom" : "fallback"))" begin
        for scenario in scenarios
            X, Y = get_input_type(scenario), get_output_type(scenario)
            @testset "$X -> $Y" begin
                (; f, x, y, der_true) = scenario
                y_out, der_out = value_and_derivative(backend, f, x)

                @testset "Primal output" begin
                    @test y_out ≈ y
                end
                @testset "Derivative output" begin
                    @test der_out ≈ der_true rtol = 1e-3
                end
                if allocs
                    @testset "Allocations" begin
                        @test (@allocated value_and_derivative(backend, f, x)) == 0
                    end
                end
                if type_stability
                    @testset "Type stability" begin
                        @test_opt value_and_derivative(backend, f, x)
                    end
                end
            end
        end
    end
end

function test_multiderivative(
    backend::AbstractBackend{custom},
    scenarios::Vector{<:Scenario}=scenarios;
    allocs::Bool=false,
    type_stability::Bool=true,
) where {custom}
    scenarios = filter(scenarios) do s
        (get_input_type(s) <: Number) && (get_output_type(s) <: AbstractArray)
    end
    @testset "Multiderivative ($(custom ? "custom" : "fallback"))" begin
        for scenario in scenarios
            X, Y = get_input_type(scenario), get_output_type(scenario)
            @testset "$X -> $Y" begin
                (; f, x, y, multider_true) = scenario
                y_out, multider_out = value_and_multiderivative(backend, f, x)

                @testset "Primal output" begin
                    @test y_out ≈ y
                end
                @testset "Multiderivative output" begin
                    @test multider_out ≈ multider_true rtol = 1e-3
                end
                @testset "Mutation" begin
                    multider_in = similar(multider_out)
                    value_and_multiderivative!(multider_in, backend, f, x)
                    @test multider_in ≈ multider_true rtol = 1e-3
                end
                if allocs
                    @testset "Allocations" begin
                        @test (@allocated value_and_multiderivative!(
                            multider_out, backend, f, x
                        )) == 0
                    end
                end
                if type_stability
                    @testset "Type stability" begin
                        @test_opt value_and_multiderivative!(multider_out, backend, f, x)
                    end
                end
            end
        end
    end
end

function test_gradient(
    backend::AbstractBackend{custom},
    scenarios::Vector{<:Scenario}=scenarios;
    allocs::Bool=false,
    type_stability::Bool=true,
) where {custom}
    scenarios = filter(scenarios) do s
        (get_input_type(s) <: AbstractArray) && (get_output_type(s) <: Number)
    end
    @testset "Gradient ($(custom ? "custom" : "fallback"))" begin
        for scenario in scenarios
            X, Y = get_input_type(scenario), get_output_type(scenario)
            @testset "$X -> $Y" begin
                (; f, x, y, grad_true) = scenario
                y_out, grad_out = value_and_gradient(backend, f, x)

                @testset "Primal output" begin
                    @test y_out ≈ y
                end
                @testset "Gradient output" begin
                    @test grad_out ≈ grad_true rtol = 1e-3
                end
                @testset "Mutation" begin
                    grad_in = similar(grad_out)
                    value_and_gradient!(grad_in, backend, f, x)
                    @test grad_in ≈ grad_true rtol = 1e-3
                end
                if allocs
                    @testset "Allocations" begin
                        @test (@allocated value_and_gradient!(grad_out, backend, f, x)) == 0
                    end
                end
                if type_stability
                    @testset "Type stability" begin
                        @test_opt value_and_gradient!(grad_out, backend, f, x)
                    end
                end
            end
        end
    end
end

function test_jacobian(
    backend::AbstractBackend{custom},
    scenarios::Vector{<:Scenario}=scenarios;
    allocs::Bool=false,
    type_stability::Bool=true,
) where {custom}
    scenarios = filter(scenarios) do s
        (get_input_type(s) <: AbstractArray) && (get_output_type(s) <: AbstractArray)
    end
    @testset "Jacobian ($(custom ? "custom" : "fallback"))" begin
        for scenario in scenarios
            X, Y = get_input_type(scenario), get_output_type(scenario)
            @testset "$X -> $Y" begin
                (; f, x, y, jac_true) = scenario
                y_out, jac_out = value_and_jacobian(backend, f, x)

                @testset "Primal output" begin
                    @test y_out ≈ y
                end
                @testset "Jacobian output" begin
                    @test jac_out ≈ jac_true rtol = 1e-3
                end
                @testset "Mutation" begin
                    jac_in = similar(jac_out)
                    value_and_jacobian!(jac_in, backend, f, x)
                    @test jac_in ≈ jac_true rtol = 1e-3
                end
                if allocs
                    @testset "Allocations" begin
                        @test (@allocated value_and_jacobian!(jac_out, backend, f, x)) == 0
                    end
                end
                if type_stability
                    @testset "Type stability" begin
                        @test_opt value_and_jacobian!(jac_out, backend, f, x)
                    end
                end
            end
        end
    end
end

function test_jacobian_and_friends(
    backend::AbstractBackend{custom},
    scenarios::Vector{<:Scenario}=scenarios;
    input_type::Type=Any,
    output_type::Type=Any,
    allocs::Bool=false,
    type_stability::Bool=true,
) where {custom}
    scenarios = filter(scenarios) do s
        (get_input_type(s) <: input_type) && (get_output_type(s) <: output_type)
    end
    test_derivative(backend, scenarios; allocs, type_stability)
    test_multiderivative(backend, scenarios; allocs, type_stability)
    test_gradient(backend, scenarios; allocs, type_stability)
    test_jacobian(backend, scenarios; allocs, type_stability)
    return nothing
end
