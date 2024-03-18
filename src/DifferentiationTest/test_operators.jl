function test_correctness(args...; kwargs...)
    return error(
        "Please load ForwardDiff.jl, this functionality is in a package extension."
    )
end

function test_type_stability(args...; kwargs...)
    return error("Please load JET.jl, this functionality is in a package extension.")
end

function test_allocations(args...; kwargs...)
    return error("Please load Chairmarks.jl, this functionality is in a package extension.")
end

const FIRST_ORDER_OPERATORS = [
    :pushforward_allocating,
    :pushforward_mutating,
    :pullback_allocating,
    :pullback_mutating,
    :derivative_allocating,
    :multiderivative_allocating,
    :multiderivative_mutating,
    :gradient_allocating,
    :jacobian_allocating,
    :jacobian_mutating,
]

const SECOND_ORDER_OPERATORS = [
    :second_derivative_allocating, :hessian_vector_product_allocating, :hessian_allocating
]

"""
$(TYPEDSIGNATURES)

Cross-test a set of `backends` for a set of `operators` on a set of `scenarios.`
"""
function test_operators(
    backends::Vector{<:AbstractADType},
    operators=vcat(FIRST_ORDER_OPERATORS, SECOND_ORDER_OPERATORS),
    scenarios::Vector{<:Scenario}=default_scenarios();
    correctness::Bool=true,
    type_stability::Bool=true,
    allocations::Bool=false,
    input_type::Type=Any,
    output_type::Type=Any,
    first_order=true,
    second_order=true,
    allocating=true,
    mutating=true,
    excluded::Vector{Symbol}=Symbol[],
)
    scenarios = filter(scenarios) do scen
        typeof(scen.x) <: input_type && typeof(scen.y) <: output_type
    end

    if !first_order
        operators = setdiff(operators, FIRST_ORDER_OPERATORS)
    end
    if !second_order
        operators = setdiff(operators, SECOND_ORDER_OPERATORS)
    end
    if !allocating
        operators = filter(op -> !endswith(string(op), "allocating"), operators)
    end
    if !mutating
        operators = filter(op -> !endswith(string(op), "mutating"), operators)
    end
    operators = filter(op -> !in(op, excluded), operators)

    @testset verbose = true "Backend tests" begin
        if correctness
            test_correctness(backends, operators, scenarios)
        end
        if type_stability
            test_type_stability(backends, operators, scenarios)
        end
        if allocations
            test_allocations(backends, operators, scenarios)
        end
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Shortcut for a single backend.
"""
function test_operators(backend::AbstractADType, args...; kwargs...)
    return test_operators([backend], args...; kwargs...)
end
