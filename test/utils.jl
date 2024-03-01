using DifferentiationInterface
using DifferentiationInterface: AbstractReverseBackend, AbstractForwardBackend
using ForwardDiff: ForwardDiff
using LinearAlgebra
using JET
using Random: AbstractRNG, randn!
using StableRNGs
using Test

## Test scenarios

struct Scenario{F,X,Y,J}
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
    "Jacobian result"
    jac_true::J
end

## Constructors

function Scenario(rng::AbstractRNG, f, x)
    y = f(x)
    return Scenario(rng, f, x, y)
end

function Scenario(rng::AbstractRNG, f::F, x::X, y::Y) where {F,X<:Number,Y<:Number}
    dx = randn(rng, X)
    dy = randn(rng, Y)
    der = ForwardDiff.derivative(f, x)
    dx_true = der * dy
    dy_true = der * dx
    jac_true = [der;;]
    return Scenario(f, x, y, dx, dy, dx_true, dy_true, jac_true)
end

function Scenario(rng::AbstractRNG, f::F, x::X, y::Y) where {F,X<:Number,Y<:AbstractArray}
    dx = randn(rng, X)
    dy = similar(y)
    randn!(rng, dy)
    der_array = ForwardDiff.derivative(f, x)
    dx_true = dot(der_array, dy)
    dy_true = der_array .* dx
    jac_true = reshape(der_array, :, 1)
    return Scenario(f, x, y, dx, dy, dx_true, dy_true, jac_true)
end

function Scenario(rng::AbstractRNG, f::F, x::X, y::Y) where {F,X<:AbstractArray,Y<:Number}
    dx = similar(x)
    randn!(rng, dx)
    dy = randn(rng, Y)
    grad = ForwardDiff.gradient(f, x)
    dx_true = grad .* dy
    dy_true = dot(grad, dx)
    jac_true = reshape(grad, 1, :)
    return Scenario(f, x, y, dx, dy, dx_true, dy_true, jac_true)
end

function Scenario(
    rng::AbstractRNG, f::F, x::X, y::Y
) where {F,X<:AbstractArray,Y<:AbstractArray}
    dx = similar(x)
    randn!(rng, dx)
    dy = similar(y)
    randn!(rng, dy)
    jac = ForwardDiff.jacobian(f, x)
    dx_true = transpose(jac) * dy
    dy_true = jac * dx
    return Scenario(f, x, y, dx, dy, dx_true, dy_true, jac)
end

## Access

get_input_type(::Scenario{F,X}) where {F,X} = X
get_output_type(::Scenario{F,X,Y}) where {F,X,Y} = Y

## Seed

rng = StableRNG(63)

## Scalar input, scalar output

scenario1 = Scenario(rng, (x::Real -> sin(2x)), 1.0)

## Scalar input, vector output

scenario2 = Scenario(rng, (x::Real -> [sin(2x), cos(3x)]), 1.0)

## Vector input, scalar output

scenario3 = Scenario(rng, (x::AbstractVector -> sin(2x[1]) + cos(3x[2])), [1.0, 2.0])

## Vector input, vector output

scenario4 = Scenario(
    rng,
    (x::AbstractVector -> [sin(2x[1]), cos(3x[2]), tan(2x[1]) + tan(3x[2])]),
    [1.0, 2.0],
)

## All

scenarios = [scenario1, scenario2, scenario3, scenario4]

## Test utilities

function test_pushforward(
    backend::AbstractForwardBackend;
    scenarios::Vector{<:Scenario}=scenarios,
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
                dy_in = zero(dy_true)
                y_out, dy_out = value_and_pushforward!(dy_in, backend, f, x, dx)

                @testset "Primal output" begin
                    @testset "Correctness" begin
                        @test y_out == y
                    end
                end
                @testset "Tangent" begin
                    @testset "Correctness" begin
                        @test dy_out ≈ dy_true rtol = 1e-3
                    end
                    if ismutable(dy_in)
                        @testset "In-place mutation" begin
                            @test dy_in ≈ dy_true rtol = 1e-3
                        end
                    end
                    if allocs
                        @testset "Allocations" begin
                            @test (@allocated value_and_pushforward!(
                                dy_in, backend, f, x, dx
                            )) == 0
                        end
                    end
                    if type_stability
                        @testset "Type stability" begin
                            @test_opt value_and_pushforward!(dy_in, backend, f, x, dx)
                        end
                    end
                end
            end
        end
    end
end

function test_pullback(
    backend::AbstractReverseBackend;
    scenarios=scenarios,
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
                dx_in = zero(dx_true)
                y_out, dx_out = value_and_pullback!(dx_in, backend, f, x, dy)

                @testset "Primal output" begin
                    @testset "Correctness" begin
                        @test y_out == y
                    end
                end
                @testset "Tangent" begin
                    @testset "Correctness" begin
                        @test dx_out ≈ dx_true rtol = 1e-3
                    end
                    if ismutable(dx_in)
                        @testset "In-place mutation" begin
                            @test dx_in ≈ dx_true rtol = 1e-3
                        end
                    end
                    if allocs
                        @testset "Allocations" begin
                            @test (@allocated value_and_pullback!(
                                dx_in, backend, f, x, dy
                            )) == 0
                        end
                    end
                    if type_stability
                        @testset "Type stability" begin
                            @test_opt value_and_pullback!(dx_in, backend, f, x, dy)
                        end
                    end
                end
            end
        end
    end
end
