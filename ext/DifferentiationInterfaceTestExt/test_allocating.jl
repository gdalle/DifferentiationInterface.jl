function test_pushforward_allocating(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    correctness::Bool=true,
    type_stability::Bool=true,
    allocs::Bool=false,
)
    @testset "$(in_type(scen)) -> $(out_type(scen))" for scen in scenarios
        @testset "Extras: $(isempty(mex))" for mex in ((), (prepare_pushforward(ba, scen),))
            if correctness
                test_correctness_pushforward_allocating(ba, scen, mex...)
            end
            if type_stability
                test_type_pushforward_allocating(ba, scen, mex...)
            end
        end
    end
end

function test_pullback_allocating(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    correctness::Bool=true,
    type_stability::Bool=true,
    allocs::Bool=false,
)
    @testset "$(in_type(scen)) -> $(out_type(scen))" for scen in scenarios
        @testset "Extras: $(isempty(mex))" for mex in ((), (prepare_pullback(ba, scen),))
            if correctness
                test_correctness_pullback_allocating(ba, scen, mex...)
            end
            if type_stability
                test_type_pullback_allocating(ba, scen, mex...)
            end
        end
    end
end

function test_derivative_allocating(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    correctness::Bool=true,
    type_stability::Bool=true,
    allocs::Bool=false,
)
    @testset "$(in_type(scen)) -> $(out_type(scen))" for scen in scenarios
        @testset "Extras: $(isempty(mex))" for mex in ((), (prepare_derivative(ba, scen),))
            if correctness
                test_correctness_derivative_allocating(ba, scen, mex...)
            end
            if type_stability
                test_type_derivative_allocating(ba, scen, mex...)
            end
        end
    end
end

function test_multiderivative_allocating(
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
                test_correctness_multiderivative_allocating(ba, scen, mex...)
            end
            if type_stability
                test_type_multiderivative_allocating(ba, scen, mex...)
            end
        end
    end
end

function test_gradient_allocating(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    correctness::Bool=true,
    type_stability::Bool=true,
    allocs::Bool=false,
)
    @testset "Gradient: $(in_type(scen)) -> $(out_type(scen))" for scen in scenarios
        @testset "Extras: $(isempty(mex))" for mex in ((), (prepare_gradient(ba, scen),))
            if correctness
                test_correctness_gradient_allocating(ba, scen, mex...)
            end
            if type_stability
                test_type_gradient_allocating(ba, scen, mex...)
            end
        end
    end
end

function test_jacobian_allocating(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    correctness::Bool=true,
    type_stability::Bool=true,
    allocs::Bool=false,
)
    @testset "$(in_type(scen)) -> $(out_type(scen))" for scen in scenarios
        @testset "Extras: $(isempty(mex))" for mex in ((), (prepare_jacobian(ba, scen),))
            if correctness
                test_correctness_jacobian_allocating(ba, scen, mex...)
            end
            if type_stability
                test_type_jacobian_allocating(ba, scen, mex...)
            end
        end
    end
end

function test_second_derivative_allocating(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    correctness::Bool=true,
    type_stability::Bool=true,
    allocs::Bool=false,
)
    @testset "$(in_type(scen)) -> $(out_type(scen))" for scen in scenarios
        @testset "Extras: $(isempty(mex))" for mex in
                                               ((), (prepare_second_derivative(ba, scen),))
            if correctness
                test_correctness_second_derivative_allocating(ba, scen, mex...)
            end
            if type_stability
                test_type_second_derivative_allocating(ba, scen, mex...)
            end
        end
    end
end

function test_hessian_allocating(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario};
    correctness::Bool=true,
    type_stability::Bool=true,
    allocs::Bool=false,
)
    @testset "$(in_type(scen)) -> $(out_type(scen))" for scen in scenarios
        @testset "Extras: $(isempty(mex))" for mex in ((), (prepare_hessian(ba, scen),))
            if correctness
                test_correctness_hessian_allocating(ba, scen, mex...)
            end
            if type_stability
                test_type_hessian_allocating(ba, scen, mex...)
            end
        end
    end
end

"""
$(TYPEDSIGNATURES)
"""
function DT.test_operators_allocating(
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
        !is_mutating(scen) && in_type(scen) <: input_type && out_type(scen) <: output_type
    end

    if mode(ba) isa ForwardMode && :pushforward in kept
        @testset "Pushforward" test_pushforward_allocating(
            ba, scenarios; correctness, type_stability, allocs
        )
    end
    if mode(ba) isa ReverseMode && :pullback in kept
        @testset "Pullback" test_pullback_allocating(
            ba, scenarios; correctness, type_stability, allocs
        )
    end
    if :derivative in kept
        @testset "Derivative" test_derivative_allocating(
            ba, derivative_scenarios(scenarios); correctness, type_stability, allocs
        )
    end
    if :multiderivative in kept
        @testset "Multiderivative" test_multiderivative_allocating(
            ba, multiderivative_scenarios(scenarios); correctness, type_stability, allocs
        )
    end
    if :gradient in kept
        @testset "Gradient" test_gradient_allocating(
            ba, gradient_scenarios(scenarios); correctness, type_stability, allocs
        )
    end
    if :jacobian in kept
        @testset "Jacobian" test_jacobian_allocating(
            ba, jacobian_scenarios(scenarios); correctness, type_stability, allocs
        )
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function DT.test_second_order_operators_allocating(
    ba::AbstractADType,
    scenarios::Vector{<:Scenario}=default_scenarios();
    input_type::Type=Any,
    output_type::Type=Any,
    correctness::Bool=true,
    type_stability::Bool=true,
    allocs::Bool=false,
    included::Vector{Symbol}=[:second_derivative, :hessian],
    excluded::Vector{Symbol}=Symbol[],
)
    kept = symdiff(included, excluded)
    scenarios = filter(scenarios) do scen
        !is_mutating(scen) && in_type(scen) <: input_type && out_type(scen) <: output_type
    end

    if :second_derivative in kept
        @testset "Second derivative" test_second_derivative_allocating(
            ba, second_derivative_scenarios(scenarios); correctness, type_stability, allocs
        )
    end
    if :hessian in kept
        @testset "Hessian" test_hessian_allocating(
            ba, hessian_scenarios(scenarios); correctness, type_stability, allocs
        )
    end
    return nothing
end
