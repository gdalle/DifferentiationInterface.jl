
function DT.test_pushforward(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    input_type::Type=Any,
    output_type::Type=Any,
    correctness::Bool=true,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        in_type(s) <: input_type && out_type(s) <: output_type
    end
    @testset "Pushforward $(in_type(scen)) -> $(out_type(scen))" for scen in scenarios
        (; f, x, y, dx, dy_true) = scen
        extras = prepare_pushforward(ba, f, x)
        @testset "Extras: $(isempty(maybe_extras))" for maybe_extras in ((), (extras,))
            y_out1, dy_out1 = value_and_pushforward(ba, f, x, dx, maybe_extras...)
            dy_in2 = zero(dy_out1)
            y_out2, dy_out2 = value_and_pushforward!(dy_in2, ba, f, x, dx, maybe_extras...)

            dy_out3 = pushforward(ba, f, x, dx, maybe_extras...)
            dy_in4 = zero(dy_out3)
            dy_out4 = pushforward!(dy_in4, ba, f, x, dx, maybe_extras...)

            if correctness
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
            end
            if type_stability
                @testset "Type stability" begin
                    @test_opt value_and_pushforward!(dy_in2, ba, f, x, dx, maybe_extras...)
                    @test_opt pushforward!(dy_in4, ba, f, x, dx, maybe_extras...)
                    @test_opt value_and_pushforward(ba, f, x, dx, maybe_extras...)
                    @test_opt pushforward(ba, f, x, dx, maybe_extras...)
                end
            end
        end
    end
end

function DT.test_pullback(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    input_type::Type=Any,
    output_type::Type=Any,
    correctness::Bool=true,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        (in_type(s) <: input_type) && (out_type(s) <: output_type)
    end
    @testset "Pullback $(in_type(scen)) -> $(out_type(scen))" for scen in scenarios
        (; f, x, y, dy, dx_true) = scen
        extras = prepare_pullback(ba, f, x)
        @testset "Extras: $(isempty(maybe_extras))" for maybe_extras in ((), (extras,))
            y_out1, dx_out1 = value_and_pullback(ba, f, x, dy, maybe_extras...)
            dx_in2 = zero(dx_out1)
            y_out2, dx_out2 = value_and_pullback!(dx_in2, ba, f, x, dy, maybe_extras...)

            dx_out3 = pullback(ba, f, x, dy, maybe_extras...)
            dx_in4 = zero(dx_out3)
            dx_out4 = pullback!(dx_in4, ba, f, x, dy, maybe_extras...)

            if correctness
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
            end
            if type_stability
                @testset "Type stability" begin
                    @test_opt value_and_pullback!(dx_in2, ba, f, x, dy, maybe_extras...)
                    @test_opt pullback!(dx_in4, ba, f, x, dy, maybe_extras...)
                    @test_opt value_and_pullback(ba, f, x, dy, maybe_extras...)
                    @test_opt pullback(ba, f, x, dy, maybe_extras...)
                end
            end
        end
    end
end

function DT.test_derivative(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    input_type::Type=Number,
    output_type::Type=Number,
    correctness::Bool=true,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        in_type(s) <: typeintersect(input_type, Number) &&
            out_type(s) <: typeintersect(output_type, Number)
    end
    @testset "Derivative $(in_type(scen)) -> $(out_type(scen))" for scen in scenarios
        (; f, x, y, der_true) = scen
        extras = prepare_derivative(ba, f, x)
        @testset "Extras: $(isempty(maybe_extras))" for maybe_extras in ((), (extras,))
            y_out1, der_out1 = value_and_derivative(ba, f, x, maybe_extras...)

            der_out2 = derivative(ba, f, x, maybe_extras...)

            if correctness
                @testset "Primal value" begin
                    @test y_out1 ≈ y
                end
                @testset "Derivative value" begin
                    @test der_out1 ≈ der_true rtol = 1e-3
                    @test der_out2 ≈ der_true rtol = 1e-3
                end
            end
            if type_stability
                @testset "Type stability" begin
                    @test_opt value_and_derivative(ba, f, x, maybe_extras...)
                    @test_opt derivative(ba, f, x, maybe_extras...)
                end
            end
        end
    end
end

function DT.test_multiderivative(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    input_type::Type=Number,
    output_type::Type=AbstractArray,
    correctness::Bool=true,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        in_type(s) <: typeintersect(input_type, Number) &&
            out_type(s) <: typeintersect(output_type, AbstractArray)
    end
    @testset "Multiderivative $(in_type(scen)) -> $(out_type(scen))" for scen in scenarios
        (; f, x, y, multider_true) = scen
        extras = prepare_multiderivative(ba, f, x)
        @testset "Extras: $(isempty(maybe_extras))" for maybe_extras in ((), (extras,))
            y_out1, multider_out1 = value_and_multiderivative(ba, f, x, maybe_extras...)
            multider_in2 = zero(multider_out1)
            y_out2, multider_out2 = value_and_multiderivative!(
                multider_in2, ba, f, x, maybe_extras...
            )

            multider_out3 = multiderivative(ba, f, x, maybe_extras...)
            multider_in4 = zero(multider_out3)
            multider_out4 = multiderivative!(multider_in4, ba, f, x, maybe_extras...)

            if correctness
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
            end
            if type_stability
                @testset "Type stability" begin
                    @test_opt value_and_multiderivative!(
                        multider_in2, ba, f, x, maybe_extras...
                    )
                    @test_opt multiderivative!(multider_in4, ba, f, x, maybe_extras...)
                    @test_opt value_and_multiderivative(ba, f, x, maybe_extras...)
                    @test_opt multiderivative(ba, f, x, maybe_extras...)
                end
            end
        end
    end
end

function DT.test_gradient(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    input_type::Type=AbstractArray,
    output_type::Type=Number,
    correctness::Bool=true,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        in_type(s) <: typeintersect(input_type, AbstractArray) &&
            out_type(s) <: typeintersect(output_type, Number)
    end
    @testset "Gradient $(in_type(scen)) -> $(out_type(scen))" for scen in scenarios
        (; f, x, y, grad_true) = scen
        extras = prepare_gradient(ba, f, x)
        @testset "Extras: $(isempty(maybe_extras))" for maybe_extras in ((), (extras,))
            y_out1, grad_out1 = value_and_gradient(ba, f, x, maybe_extras...)
            grad_in2 = zero(grad_out1)
            y_out2, grad_out2 = value_and_gradient!(grad_in2, ba, f, x, maybe_extras...)

            grad_out3 = gradient(ba, f, x, maybe_extras...)
            grad_in4 = zero(grad_out3)
            grad_out4 = gradient!(grad_in4, ba, f, x, maybe_extras...)

            if correctness
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
            end
            if type_stability
                @testset "Type stability" begin
                    @test_opt value_and_gradient!(grad_in2, ba, f, x, maybe_extras...)
                    @test_opt gradient!(grad_in4, ba, f, x, maybe_extras...)
                    @test_opt value_and_gradient(ba, f, x, maybe_extras...)
                    @test_opt gradient(ba, f, x, maybe_extras...)
                end
            end
        end
    end
end

function DT.test_jacobian(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    input_type::Type=AbstractArray,
    output_type::Type=AbstractArray,
    correctness::Bool=true,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        in_type(s) <: typeintersect(input_type, AbstractArray) &&
            out_type(s) <: typeintersect(output_type, AbstractArray)
    end
    @testset "Jacobian $(in_type(scen)) -> $(out_type(scen))" for scen in scenarios
        (; f, x, y, jac_true) = scen
        extras = prepare_jacobian(ba, f, x)
        @testset "Extras: $(isempty(maybe_extras))" for maybe_extras in ((), (extras,))
            y_out1, jac_out1 = value_and_jacobian(ba, f, x, maybe_extras...)
            jac_in2 = zero(jac_out1)
            y_out2, jac_out2 = value_and_jacobian!(jac_in2, ba, f, x, maybe_extras...)

            jac_out3 = jacobian(ba, f, x, maybe_extras...)
            jac_in4 = zero(jac_out3)
            jac_out4 = jacobian!(jac_in4, ba, f, x, maybe_extras...)

            if correctness
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
            end
            if type_stability
                @testset "Type stability" begin
                    @test_opt value_and_jacobian!(jac_in2, ba, f, x, maybe_extras...)
                    @test_opt jacobian!(jac_in4, ba, f, x, maybe_extras...)
                    @test_opt value_and_jacobian(ba, f, x, maybe_extras...)
                    @test_opt jacobian(ba, f, x, maybe_extras...)
                end
            end
        end
    end
end

function DT.test_all_operators(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    input_type::Type=Any,
    output_type::Type=Any,
    correctness::Bool=true,
    type_stability::Bool=true,
)
    if autodiff_mode(ba) isa ForwardMode
        @testset "Pushforward" test_pushforward(
            ba, scenarios; input_type, output_type, correctness, type_stability
        )
    elseif autodiff_mode(ba) isa ReverseMode
        @testset "Pullback" test_pullback(
            ba, scenarios; input_type, output_type, correctness, type_stability
        )
    end
    @testset "Derivative" test_derivative(
        ba, scenarios; input_type, output_type, correctness, type_stability
    )
    @testset "Multiderivative" test_multiderivative(
        ba, scenarios; input_type, output_type, correctness, type_stability
    )
    @testset "Gradient" test_gradient(
        ba, scenarios; input_type, output_type, correctness, type_stability
    )
    @testset "Jacobian" test_jacobian(
        ba, scenarios; input_type, output_type, correctness, type_stability
    )
    return nothing
end
