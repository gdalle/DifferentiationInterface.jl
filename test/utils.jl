using DifferentiationInterface
using JET
using Test

## Test scenarios

@kwdef struct Scenario{F,X,Y}
    "function"
    f::F
    "argument"
    x::X
    "pushforward seed"
    dx::X
    "pushforward result"
    dy_true::Y
    "pullback seed"
    dy::Y
    "pullback result"
    dx_true::X
end

get_input_type(::Scenario{F,X}) where {F,X} = X
get_output_type(::Scenario{F,X,Y}) where {F,X,Y} = Y

## Scalar input, scalar output

scenario1 = Scenario(;
    f=(x::Real -> exp(2x)), x=1.0, dx=5.0, dy_true=2exp(2) * 5, dy=5.0, dx_true=2exp(2) * 5
)

## Scalar input, vector output

scenario2 = Scenario(;
    f=(x::Real -> [exp(2x), exp(3x)]),
    x=1.0,
    dx=5.0,
    dy_true=[2exp(2), 3exp(3)] .* 5,
    dy=[0.0, 5.0],
    dx_true=3exp(3) * 5,
)

## Vector input, scalar output

scenario3 = Scenario(;
    f=(x::AbstractVector -> exp(2x[1]) + exp(3x[2])),
    x=[1.0, 2.0],
    dx=[0.0, 5.0],
    dy_true=3exp(6) * 5,
    dy=5.0,
    dx_true=[2exp(2), 3exp(6)] .* 5,
)

## Vector input, vector output

scenario4 = Scenario(;
    f=(x::AbstractVector -> [exp(2x[1]), exp(3x[2])]),
    x=[1.0, 2.0],
    dx=[0.0, 5.0],
    dy_true=[0.0, 3exp(6)] .* 5,
    dy=[0.0, 5.0],
    dx_true=[0.0, 3exp(6)] .* 5,
)

## All

scenarios = [scenario1, scenario2, scenario3, scenario4]

## Test utilities

function test_pushforward(
    backend;
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
                (; f, x, dx, dy_true) = scenario
                @test pushforward!(zero(dy_true), backend, f, x, dx) ≈ dy_true rtol = 1e-3
                if allocs
                    dy = zero(dy_true)
                    @test (@allocated pushforward!(dy, backend, f, x, dx)) == 0
                end
                if type_stability
                    @test_opt pushforward!(zero(dy_true), backend, f, x, dx)
                end
            end
        end
    end
end

function test_pullback(
    backend;
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
                (; f, x, dy, dx_true) = scenario

                dx_in = zero(dx_true)
                dx_out = pullback!(dx_in, backend, f, x, dy)

                # Test output
                @test dx_out ≈ dx_true rtol = 1e-3
                # Test in-place mutation
                if ismutable(dx_in)
                    @test dx_in ≈ dx_true rtol = 1e-3
                end
                if allocs
                    @test (@allocated pullback!(dx_in, backend, f, x, dy)) == 0
                end
                if type_stability
                    @test_opt pullback!(dx_in, backend, f, x, dy)
                end
            end
        end
    end
end
