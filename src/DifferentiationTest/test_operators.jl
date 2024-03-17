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

const SECOND_ORDER_OPERATORS = [:second_derivative_allocating, :hessian_allocating]

"""
$(TYPEDSIGNATURES)
"""
function test_operators(
    backend::AbstractADType,
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
    operators=vcat(FIRST_ORDER_OPERATORS, SECOND_ORDER_OPERATORS),
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
            @testset verbose = true "Correctness" begin
                test_correctness(backend, scenarios; operators)
            end
        end
        if type_stability
            @testset verbose = true "Type stability" begin
                test_type_stability(backend, scenarios; operators)
            end
        end
        if allocations
            @testset verbose = true "Allocations" begin
                test_allocations(backend, scenarios; operators)
            end
        end
    end
    return nothing
end
