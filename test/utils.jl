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
    jac_true = der
    return Scenario(f, x, y, dx, dy, dx_true, dy_true, jac_true)
end

function Scenario(rng::AbstractRNG, f::F, x::X, y::Y) where {F,X<:Number,Y<:AbstractArray}
    dx = randn(rng, X)
    dy = similar(y)
    randn!(rng, dy)
    der_array = ForwardDiff.derivative(f, x)
    dx_true = dot(der_array, dy)
    dy_true = der_array .* dx
    jac_true = der_array
    return Scenario(f, x, y, dx, dy, dx_true, dy_true, jac_true)
end

function Scenario(rng::AbstractRNG, f::F, x::X, y::Y) where {F,X<:AbstractArray,Y<:Number}
    dx = similar(x)
    randn!(rng, dx)
    dy = randn(rng, Y)
    grad = ForwardDiff.gradient(f, x)
    dx_true = grad .* dy
    dy_true = dot(grad, dx)
    jac_true = grad
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

scenario_scalar_scalar = Scenario(rng, (x::Number -> sin(x)::Number), 1.0)

scenario_scalar_vector = Scenario(
    rng, (x::Number -> [sin(x), sin(2x)]::AbstractVector), 1.0
)

scenario_scalar_matrix = Scenario(
    rng, (x::Number -> [sin(x) cos(x); sin(2x) cos(2x)]::AbstractMatrix), 1.0
)

scenario_vector_scalar = Scenario(
    rng, (x::AbstractVector -> sum(sin(i * x[i]) for i in eachindex(x))::Number), [1.0, 2.0]
)

scenario_matrix_scalar = Scenario(
    rng,
    (
        x::AbstractMatrix -> sum(
            sin(i * x[i, j]) + cos(j * x[i, j]) for i in axes(x, 1) for j in axes(x, 2)
        )::Number
    ),
    [1.0 2.0; 3.0 4.0],
)

scenario_vector_vector = Scenario(
    rng,
    (
        x::AbstractVector -> vcat(
            [sin(i * x[i]) for i in eachindex(x)], #
            [cos(i * x[i]) for i in eachindex(x)],
        )::AbstractVector
    ),
    [1.0, 2.0],
)

scenario_vector_matrix = Scenario(
    rng,
    (
        x::AbstractVector -> hcat(
            [sin(i * x[i]) for i in eachindex(x)], #
            [cos(i * x[i]) for i in eachindex(x)],
        )::AbstractMatrix
    ),
    [1.0, 2.0],
)

scenario_matrix_vector = Scenario(
    rng,
    (
        x::AbstractVector -> vcat(
            [sin(i * x[i, j]) for i in axes(x, 1) for j in axes(x, 2)],
            [cos(j * x[i, j]) for i in axes(x, 1) for j in axes(x, 2)],
        )::AbstractVector
    ),
    [1.0 2.0; 3.0 4.0],
)

scenario_matrix_matrix = Scenario(
    rng,
    (
        x::AbstractVector -> hcat(
            [sin(i * x[i, j]) for i in axes(x, 1) for j in axes(x, 2)],
            [cos(j * x[i, j]) for i in axes(x, 1) for j in axes(x, 2)],
        )::AbstractMatrix
    ),
    [1.0 2.0; 3.0 4.0],
)

## All

scenarios = [
    scenario_scalar_scalar,
    scenario_scalar_vector,
    scenario_scalar_matrix,
    scenario_vector_scalar,
    scenario_matrix_scalar,
    scenario_vector_vector,
    scenario_vector_matrix,
    scenario_matrix_vector,
    scenario_matrix_matrix,
]

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

function test_jacobian(
    backend;
    scenarios::Vector{<:Scenario}=scenarios,
    input_type::Type=Any,
    output_type::Type=Any,
    allocs::Bool=false,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        (get_input_type(s) <: input_type) && (get_output_type(s) <: output_type)
    end
    @testset "Jacobian" begin
        for scenario in scenarios
            X, Y = get_input_type(scenario), get_output_type(scenario)
            @testset "$X -> $Y" begin
                (; f, x, y, jac_true) = scenario
                y_out, jac_out = value_and_jacobian(backend, f, x)

                @testset "Allocation of Jacobian" begin
                    @testset "Primal output" begin
                        @test y_out == y
                    end
                    @testset "Jacobian output" begin
                        @test jac_out ≈ jac_true
                    end
                end
                @testset "In-place mutation" begin
                    fill!(jac_out, 42)
                    y_out, jac_out = value_and_jacobian!(jac_out, backend, f, x)
                    @testset "Primal output" begin
                        @test y_out == y
                    end
                    @testset "Jacobian output" begin
                        @test jac_out ≈ jac_true
                    end

                    if allocs
                        @testset "Allocations" begin
                            @test (@allocated value_and_jacobian!(
                                jac_out, backend, f, x
                            )) == 0
                        end
                    end
                    if type_stability
                        @testset "Type stability" begin
                            @test_opt value_and_jacobian!(jac_out, backend, f, x)
                        end
                    end
                end
            end # scenario testset
        end # scenario loop
    end # Jacobian testset
end
