function test_pushforward_mutating(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    correctness::Bool=true,
    type_stability::Bool=true,
    allocs::Bool=false,
)
    @testset "$(in_type(scen)) -> $(out_type(scen))" for scen in scenarios
        @testset "Extras: $(isempty(mex))" for mex in ((), (prepare_pushforward(ba, scen),))
            if correctness
                test_correctness_pushforward_mutating(ba, scen, mex...)
            end
            if type_stability
                test_type_pushforward_mutating(ba, scen, mex...)
            end
        end
    end
end

function test_pullback_mutating(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario}=default_scenarios();
    correctness::Bool=true,
    type_stability::Bool=true,
    allocs::Bool=false,
)
    @testset "$(in_type(scen)) -> $(out_type(scen))" for scen in scenarios
        @testset "Extras: $(isempty(mex))" for mex in ((), (prepare_pullback(ba, scen),))
            if correctness
                test_correctness_pullback_mutating(ba, scen, mex...)
            end
            if type_stability
                test_type_pullback_mutating(ba, scen, mex...)
            end
        end
    end
end

function test_multiderivative_mutating(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    correctness::Bool=true,
    type_stability::Bool=true,
    allocs::Bool=false,
)
    @testset "$(in_type(scen)) -> $(out_type(scen))" for scen in scenarios
        @testset "Extras: $(isempty(mex))" for mex in
                                               ((), (prepare_multiderivative(ba, scen),))
            if correctness
                test_correctness_multiderivative_mutating(ba, scen, mex...)
            end
            if type_stability
                test_type_multiderivative_mutating(ba, scen, mex...)
            end
        end
    end
end

function test_jacobian_mutating(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    correctness::Bool=true,
    type_stability::Bool=true,
    allocs::Bool=false,
)
    @testset "$(in_type(scen)) -> $(out_type(scen))" for scen in scenarios
        @testset "Extras: $(isempty(mex))" for mex in ((), (prepare_jacobian(ba, scen),))
            if correctness
                test_correctness_jacobian_mutating(ba, scen, mex...)
            end
            if type_stability
                test_type_jacobian_mutating(ba, scen, mex...)
            end
        end
    end
end

"""
$(TYPEDSIGNATURES)
"""
function DT.test_operators_mutating(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario}=default_scenarios();
    input_type::Type=Any,
    output_type::Type=Any,
    correctness::Bool=true,
    type_stability::Bool=true,
    allocs::Bool=false,
    included::Vector{Symbol}=[
        :pushforward, :pullback, :derivative, :multiderivative, :gradient, :jacobian
    ],
    excluded::Vector{Symbol}=Symbol[],
)
    kept = symdiff(included, excluded)
    scenarios = filter(scenarios) do scen
        is_mutating(scen) && in_type(scen) <: input_type && out_type(scen) <: output_type
    end

    if autodiff_mode(ba) isa ForwardMode && :pushforward in kept
        @testset "Pushforward (mutating)" test_pushforward_mutating(
            ba, scenarios; correctness, type_stability, allocs
        )
    end
    if autodiff_mode(ba) isa ReverseMode && :pullback in kept
        @testset "Pullback (mutating)" test_pullback_mutating(
            ba, scenarios; correctness, type_stability, allocs
        )
    end
    if :multiderivative in kept
        @testset "Multiderivative (mutating)" test_multiderivative_mutating(
            ba, multiderivative_scenarios(scenarios); correctness, type_stability, allocs
        )
    end
    if :jacobian in kept
        @testset "Jacobian (mutating)" test_jacobian_mutating(
            ba, jacobian_scenarios(scenarios); correctness, type_stability, allocs
        )
    end
    return nothing
end
