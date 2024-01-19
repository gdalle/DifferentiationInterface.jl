using DifferentiationInterface
using JET
using Test

fa(x::AbstractVector) = exp(2x[1]) + exp(3x[2])  # vector to scalar
fb(x::AbstractVector) = [exp(2x[1]), exp(3x[2])]  # vector to vector

x = [1.0, 2.0]

## JVP

dx = [0.0, 5.0]
dya_true = 3exp(3x[2]) * dx[2]
dyb_true = [0.0, 3exp(3x[2])] .* dx[2]

## VJP

dya = 5.0
dxa_true = [2exp(2x[1]), 3exp(3x[2])] .* dya

dyb = [0.0, 5.0]
dxb_true = [0.0, 3exp(3x[2])] .* dyb[2]

## Tests

@kwdef struct Scenario{F,X,Y}
    f::F
    x::X
    dx::X
    dy::Y
    dx_true::X
    dy_true::Y
end

get_input_type(::Scenario{F,X}) where {F,X} = X
get_output_type(::Scenario{F,X,Y}) where {F,X,Y} = Y

scenarios = (
    Scenario(fa, x, dx, dya, dxa_true, dya_true),
    Scenario(fb, x, dx, dyb, dxb_true, dyb_true),
)

function test_pushforward(
    backend;
    scenarios=scenarios,
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
            (; f, x, dx, dy_true) = scenario
            @test pushforward!(zero(dy_true), backend, f, x, dx) ≈ dy_true
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
            (; f, x, dy, dx_true) = scenario
            @test pullback!(zero(dx_true), backend, f, x, dy) ≈ dx_true
            if allocs
                dx = zero(dx_true)
                @test (@allocated pullback!(dx, backend, f, x, dy)) == 0
            end
            if type_stability
                @test_opt pullback!(zero(dx_true), backend, f, x, dy)
            end
        end
    end
end
