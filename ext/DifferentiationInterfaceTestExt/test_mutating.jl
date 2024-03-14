function DT.test_pushforward_mutating(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    input_type::Type=Any,
    output_type::Type=Any,
    correctness::Bool=true,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        in_type(s) <: input_type && out_type(s) <: output_type && mutating(s)
    end
    @testset "Pushforward (mutating): $(in_type(scen)) -> $(out_type(scen))" for scen in
                                                                                 scenarios
        (; f, x, y, dx, dy_true) = scen
        f! = f
        extras = prepare_pushforward(ba, f!, x, y)
        @testset "Extras: $(isempty(maybe_extras))" for maybe_extras in ((), (extras,))
            y_in = zero(y)
            dy_in = zero(dy_true)
            y_out, dy_out = value_and_pushforward!(
                y_in, dy_in, ba, f!, x, dx, maybe_extras...
            )

            if correctness
                @testset "Primal value" begin
                    @test y_out ≈ y
                    @testset "Mutation" begin
                        @test y_in ≈ y
                    end
                end
                @testset "Tangent value" begin
                    @test dy_out ≈ dy_true rtol = 1e-3
                    @testset "Mutation" begin
                        @test dy_in ≈ dy_true rtol = 1e-3
                    end
                end
            end
            if type_stability
                @testset "Type stability" begin
                    @test_opt value_and_pushforward!(
                        y_in, dy_in, ba, f!, x, dx, maybe_extras...
                    )
                end
            end
        end
    end
end

function DT.test_pullback_mutating(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    input_type::Type=Any,
    output_type::Type=Any,
    correctness::Bool=true,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        in_type(s) <: input_type && out_type(s) <: output_type && mutating(s)
    end
    @testset "Pullback (mutating): $(in_type(scen)) -> $(out_type(scen))" for scen in
                                                                              scenarios
        (; f, x, y, dy, dx_true) = scen
        f! = f
        extras = prepare_pullback(ba, f!, x, y)
        @testset "Extras: $(isempty(maybe_extras))" for maybe_extras in ((), (extras,))
            y_in = zero(y)
            dx_in = zero(dx_true)
            y_out, dx_out = value_and_pullback!(y_in, dx_in, ba, f!, x, dy, maybe_extras...)

            if correctness
                @testset "Primal value" begin
                    @test y_out ≈ y
                    @testset "Mutation" begin
                        @test y_in ≈ y
                    end
                end
                @testset "Cotangent value" begin
                    @test dx_out ≈ dx_true rtol = 1e-3
                    if ismutable(dx_true)
                        @testset "Mutation" begin
                            @test dx_in ≈ dx_true rtol = 1e-3
                        end
                    end
                end
            end
            if type_stability
                @testset "Type stability" begin
                    @test_opt value_and_pullback!(
                        y_in, dx_in, ba, f!, x, dy, maybe_extras...
                    )
                end
            end
        end
    end
end

function DT.test_multiderivative_mutating(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    input_type::Type=Number,
    output_type::Type=AbstractArray,
    correctness::Bool=true,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        in_type(s) <: typeintersect(input_type, Number) &&
            out_type(s) <: typeintersect(output_type, AbstractArray) &&
            mutating(s)
    end
    @testset "Multiderivative (mutating): $(in_type(scen)) -> $(out_type(scen))" for scen in
                                                                                     scenarios
        (; f, x, y, multider_true) = scen
        f! = f
        extras = prepare_multiderivative(ba, f!, x, y)
        @testset "Extras: $(isempty(maybe_extras))" for maybe_extras in ((), (extras,))
            y_in = zero(y)
            multider_in = zero(multider_true)
            y_out, multider_out = value_and_multiderivative!(
                y_in, multider_in, ba, f!, x, maybe_extras...
            )

            if correctness
                @testset "Primal value" begin
                    @test y_out ≈ y
                    @testset "Mutation" begin
                        @test y_in ≈ y
                    end
                end
                @testset "Multiderivative value" begin
                    @test multider_out ≈ multider_true rtol = 1e-3
                    @testset "Mutation" begin
                        @test multider_in ≈ multider_true rtol = 1e-3
                    end
                end
            end
            if type_stability
                @testset "Type stability" begin
                    @test_opt value_and_multiderivative!(
                        y_in, multider_in, ba, f!, x, maybe_extras...
                    )
                end
            end
        end
    end
end

function DT.test_jacobian_mutating(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    input_type::Type=AbstractArray,
    output_type::Type=AbstractArray,
    correctness::Bool=true,
    type_stability::Bool=true,
)
    scenarios = filter(scenarios) do s
        in_type(s) <: typeintersect(input_type, AbstractArray) &&
            out_type(s) <: typeintersect(output_type, AbstractArray) &&
            mutating(s)
    end
    @testset "Jacobian (mutating): $(in_type(scen)) -> $(out_type(scen))" for scen in
                                                                              scenarios
        (; f, x, y, jac_true) = scen
        f! = f
        extras = prepare_jacobian(ba, f!, x, y)
        @testset "Extras: $(isempty(maybe_extras))" for maybe_extras in ((), (extras,))
            y_in = zero(y)
            jac_in = similar(y, length(y), length(x))
            y_out, jac_out = value_and_jacobian!(y_in, jac_in, ba, f!, x, maybe_extras...)

            if correctness
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
            end
            if type_stability
                @testset "Type stability" begin
                    @test_opt value_and_jacobian!(y_in, jac_in, ba, f!, x, maybe_extras...)
                end
            end
        end
    end
end

function DT.test_all_operators_mutating(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    input_type::Type=Any,
    output_type::Type=Any,
    correctness::Bool=true,
    type_stability::Bool=true,
)
    if autodiff_mode(ba) isa ForwardMode
        @testset "Pushforward (mutating)" test_pushforward_mutating(
            ba, scenarios; input_type, output_type, correctness, type_stability
        )
    elseif autodiff_mode(ba) isa ReverseMode
        @testset "Pullback (mutating)" test_pullback_mutating(
            ba, scenarios; input_type, output_type, correctness, type_stability
        )
    end
    @testset "Multiderivative (mutating)" test_multiderivative_mutating(
        ba, scenarios; input_type, output_type, correctness, type_stability
    )
    @testset "Jacobian (mutating)" test_jacobian_mutating(
        ba, scenarios; input_type, output_type, correctness, type_stability
    )
    return nothing
end
