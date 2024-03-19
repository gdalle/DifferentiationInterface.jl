test_correctness(args...; kwargs...) = error("Please load ForwardDiff.jl")
test_type_stability(args...; kwargs...) = error("Please load JET.jl")
run_benchmark(args...; kwargs...) = error("Please load Chairmarks.jl")
test_allocations(args...; kwargs...) = error("Please load Chairmarks.jl")

"""
    parse_benchmark(result; aggregators)

Parse the output of `test_operators(args...; benchmark=true)` into a `DataFrame`.

# Keyword arguments

- `aggregators=[minimum]`: aggregation functions to apply on benchmark samples
"""
parse_benchmark(args...; kwargs...) = error("Please load Chairmarks.jl and DataFrames.jl")

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

function filter_operators(
    operators::Vector{Symbol};
    first_order::Bool,
    second_order::Bool,
    allocating::Bool,
    mutating::Bool,
    excluded::Vector{Symbol},
)
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
    return operators
end

"""
    test_operators(
        backends, [operators, scenarios];
        correctness, type_stability, benchmark, allocations,
        input_type, output_type,
        first_order, second_order, allocating, mutating,
        excluded,
    )

Cross-test a list of `backends` for a list of `operators` on a list of `scenarios.`

Return `nothing`, except when `benchmark=true`.

# Default arguments

- `operators`: defaults to all of them
- `scenarios`: defaults to a set of default scenarios

# Keyword arguments

- `correctness=true`: whether to compare the differentiation results with those given by ForwardDiff.jl
- `type_stability=true`: whether to check type stability with JET.jl
- `benchmark=false`: whether to run and return a benchmark suite with Chairmarks.jl
- `allocations=false`: whether to check that the benchmarks are allocation-free
- `input_type=Any`: restrict scenario inputs to subtypes of this
- `output_type=Any`: restrict scenario outputs to subtypes of this
- `first_order=true`: consider first order operators
- `second_order=true`: consider second order operators
- `allocating=true`: consider operators for allocating functions
- `mutating=true`: consider operators for mutating functions
- `excluded=Symbol[]`: list of excluded operators
"""
function test_operators(
    backends::Vector{<:AbstractADType},
    operators::Vector{Symbol}=vcat(FIRST_ORDER_OPERATORS, SECOND_ORDER_OPERATORS),
    scenarios::Vector{<:Scenario}=default_scenarios();
    correctness::Bool=true,
    type_stability::Bool=true,
    benchmark::Bool=false,
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
    operators = filter_operators(
        operators; first_order, second_order, allocating, mutating, excluded
    )
    result = nothing
    @testset verbose = true "Backend tests" begin
        if correctness
            test_correctness(backends, operators, scenarios)
        end
        if type_stability
            test_type_stability(backends, operators, scenarios)
        end
        if benchmark || allocations
            result = run_benchmark(
                backends, operators, scenarios; test_allocations=allocations
            )
        end
    end
    if benchmark
        return result
    else
        return nothing
    end
end

"""
$(TYPEDSIGNATURES)

Shortcut for a single backend.
"""
function test_operators(backend::AbstractADType, args...; kwargs...)
    return test_operators([backend], args...; kwargs...)
end
