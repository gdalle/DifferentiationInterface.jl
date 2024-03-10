using ADTypes: AbstractADType
using DifferentiationInterface
using DifferentiationInterface:
    AbstractImplem, CustomImplem, FallbackImplem, ForwardMode, ReverseMode, autodiff_mode
using JET
using Test

pretty(::CustomImplem) = "custom"
pretty(::FallbackImplem) = "fallback"

## Test utilities

function test_pushforward(
    backend::AbstractADType,
    scenarios::Vector{<:Scenario};
    input_type::Type=Any,
    output_type::Type=Any,
    allocs::Bool=false,
    type_stability::Bool=true,
)
    if !isa(autodiff_mode(backend), ForwardMode)
        return nothing
    end
    scenarios = filter(scenarios) do s
        get_input_type(s) <: input_type && get_output_type(s) <: output_type
    end
    @testset "Pushforward" begin
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
    backend::AbstractADType,
    scenarios::Vector{<:Scenario};
    input_type::Type=Any,
    output_type::Type=Any,
    allocs::Bool=false,
    type_stability::Bool=true,
)
    if !isa(autodiff_mode(backend), ReverseMode)
        return nothing
    end
    scenarios = filter(scenarios) do s
        (get_input_type(s) <: input_type) && (get_output_type(s) <: output_type)
    end
    @testset "Pullback" begin
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
    implem::AbstractImplem,
    backend::AbstractADType,
    scenarios::Vector{<:Scenario};
    allocs::Bool=false,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        (get_input_type(s) <: Number) && (get_output_type(s) <: Number)
    end
    @testset "Derivative ⁻ $(pretty(implem))" begin
        for scenario in scenarios
            X, Y = get_input_type(scenario), get_output_type(scenario)
            handles_types(backend, X, Y) || continue

            @testset "$X -> $Y" begin
                (; f, x, y, der_true) = scenario

                y_out1, der_out1 = value_and_derivative(implem, backend, f, x)

                der_out2 = derivative(implem, backend, f, x)

                @testset "Primal value" begin
                    @test y_out1 ≈ y
                end
                @testset "Derivative value" begin
                    @test der_out1 ≈ der_true rtol = 1e-3
                    @test der_out2 ≈ der_true rtol = 1e-3
                end
                allocs && @testset "Allocations" begin
                    @test iszero(@allocated value_and_derivative(implem, backend, f, x))
                    @test iszero(@allocated derivative(implem, backend, f, x))
                end
                type_stability && @testset "Type stability" begin
                    @test_opt value_and_derivative(implem, backend, f, x)
                    @test_opt derivative(implem, backend, f, x)
                end
            end
        end
    end
end

function test_multiderivative(
    implem::AbstractImplem,
    backend::AbstractADType,
    scenarios::Vector{<:Scenario};
    allocs::Bool=false,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        (get_input_type(s) <: Number) && (get_output_type(s) <: AbstractArray)
    end
    @testset "Multiderivative ⁻ $(pretty(implem))" begin
        for scenario in scenarios
            X, Y = get_input_type(scenario), get_output_type(scenario)
            handles_types(backend, X, Y) || continue

            @testset "$X -> $Y" begin
                (; f, x, y, multider_true) = scenario

                y_out1, multider_out1 = value_and_multiderivative(implem, backend, f, x)
                multider_in2 = zero(multider_out1)
                y_out2, multider_out2 = value_and_multiderivative!(
                    implem, multider_in2, backend, f, x
                )

                multider_out3 = multiderivative(implem, backend, f, x)
                multider_in4 = zero(multider_out3)
                multider_out4 = multiderivative!(implem, multider_in4, backend, f, x)

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
                        @allocated value_and_multiderivative!(
                            implem, multider_in2, backend, f, x
                        )
                    )
                    @test iszero(
                        @allocated multiderivative!(implem, multider_in4, backend, f, x)
                    )
                end
                type_stability && @testset "Type stability" begin
                    @test_opt value_and_multiderivative!(implem, multider_in2, backend, f, x)
                    @test_opt multiderivative!(implem, multider_in4, backend, f, x)
                end
            end
        end
    end
end

function test_gradient(
    implem::AbstractImplem,
    backend::AbstractADType,
    scenarios::Vector{<:Scenario};
    allocs::Bool=false,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        (get_input_type(s) <: AbstractArray) && (get_output_type(s) <: Number)
    end
    @testset "Gradient ⁻ $(pretty(implem))" begin
        for scenario in scenarios
            X, Y = get_input_type(scenario), get_output_type(scenario)
            handles_types(backend, X, Y) || continue

            @testset "$X -> $Y" begin
                (; f, x, y, grad_true) = scenario

                y_out1, grad_out1 = value_and_gradient(implem, backend, f, x)
                grad_in2 = zero(grad_out1)
                y_out2, grad_out2 = value_and_gradient!(implem, grad_in2, backend, f, x)

                grad_out3 = gradient(implem, backend, f, x)
                grad_in4 = zero(grad_out3)
                grad_out4 = gradient!(implem, grad_in4, backend, f, x)

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
                    @test iszero(
                        @allocated value_and_gradient!(implem, grad_in2, backend, f, x)
                    )
                    @test iszero(@allocated gradient!(implem, grad_in4, backend, f, x))
                end
                type_stability && @testset "Type stability" begin
                    @test_opt value_and_gradient!(implem, grad_in2, backend, f, x)
                    @test_opt gradient!(implem, grad_in4, backend, f, x)
                end
            end
        end
    end
end

function test_jacobian(
    implem::AbstractImplem,
    backend::AbstractADType,
    scenarios::Vector{<:Scenario};
    allocs::Bool=false,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        (get_input_type(s) <: AbstractArray) && (get_output_type(s) <: AbstractArray)
    end
    @testset "Jacobian ⁻ $(pretty(implem))" begin
        for scenario in scenarios
            X, Y = get_input_type(scenario), get_output_type(scenario)
            handles_types(backend, X, Y) || continue

            @testset "$X -> $Y" begin
                (; f, x, y, jac_true) = scenario

                y_out1, jac_out1 = value_and_jacobian(implem, backend, f, x)
                jac_in2 = zero(jac_out1)
                y_out2, jac_out2 = value_and_jacobian!(implem, jac_in2, backend, f, x)

                jac_out3 = jacobian(implem, backend, f, x)
                jac_in4 = zero(jac_out3)
                jac_out4 = jacobian!(implem, jac_in4, backend, f, x)

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
                    @test iszero(@allocated value_and_jacobian!(implem, jac_in2, backend, f, x))
                    @test iszero(@allocated jacobian!(implem, jac_in4, backend, f, x))
                end
                type_stability && @testset "Type stability" begin
                    @test_opt value_and_jacobian!(implem, jac_in2, backend, f, x)
                    @test_opt jacobian!(implem, jac_in4, backend, f, x)
                end
            end
        end
    end
end

function test_jacobian_and_friends(
    implem::AbstractImplem,
    backend::AbstractADType,
    scenarios::Vector{<:Scenario};
    input_type::Type=Any,
    output_type::Type=Any,
    allocs::Bool=false,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        (get_input_type(s) <: input_type) && (get_output_type(s) <: output_type)
    end
    test_derivative(implem, backend, scenarios; allocs, type_stability)
    test_multiderivative(implem, backend, scenarios; allocs, type_stability)
    test_gradient(implem, backend, scenarios; allocs, type_stability)
    test_jacobian(implem, backend, scenarios; allocs, type_stability)
    return nothing
end
