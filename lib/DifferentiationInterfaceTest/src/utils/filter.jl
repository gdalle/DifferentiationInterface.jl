"""
    all_operators()

List all operators that can be tested with [`test_differentiation`](@ref).
"""
function all_operators()
    return [
        pushforward,
        pullback,
        derivative,
        gradient,
        jacobian,
        second_derivative,
        hvp,
        hessian,
    ]
end

function filter_operators(
    operators::Vector{<:Function};
    first_order::Bool,
    second_order::Bool,
    excluded::Vector{<:Function},
)
    !first_order && (
        operators = filter(
            !in([pushforward, pullback, derivative, gradient, jacobian]), operators
        )
    )
    !second_order && (operators = filter(!in([second_derivative, hvp, hessian]), operators))
    operators = filter(!in(excluded), operators)
    return operators
end

function filter_scenarios(
    scenarios::Vector{<:Scenario};
    input_type::Type,
    output_type::Type,
    allocating::Bool,
    mutating::Bool,
)
    scenarios = filter(scenarios) do scen
        typeof(scen.x) <: input_type && typeof(scen.y) <: output_type
    end
    !allocating && (scenarios = filter(is_mutating, scenarios))
    !mutating && (scenarios = filter(!is_mutating, scenarios))
    return scenarios
end
