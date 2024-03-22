test_correctness(args...; kwargs...) = error("Please load ForwardDiff.jl")
test_type_stability(args...; kwargs...) = error("Please load JET.jl")

FIRST_ORDER_OPERATORS = [
    :pushforward_allocating,
    :pushforward_mutating,
    :pullback_allocating,
    :pullback_mutating,
    :derivative_allocating,
    :derivative_mutating,
    :gradient_allocating,
    :jacobian_allocating,
    :jacobian_mutating,
]

SECOND_ORDER_OPERATORS = [
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
        correctness, type_stability, call_count, benchmark, allocations,
        input_type, output_type,
        first_order, second_order, allocating, mutating,
        excluded,
    )

Cross-test a list of `backends` for a list of `operators` on a list of `scenarios`, running a variety of different tests.

- If `benchmark` is `false`, this returns a `TestSet` object.
- If `benchmark` is `true`, this returns a [`BenchmarkData`](@ref) object, which is easy to turn into a `DataFrame`.

# Default arguments

- `operators`: `$FIRST_ORDER_OPERATORS`
- `scenarios`: the output of [`default_scenarios()`](@ref)

# Keyword arguments

Test families:

- `correctness=true`: whether to compare the differentiation results with those given by ForwardDiff.jl
- `type_stability=true`: whether to check type stability with JET.jl
- `call_count=false`: whether to check that the function is called the right number of times
- `benchmark=false`: whether to run and return a benchmark suite with Chairmarks.jl
- `allocations=false`: whether to check that the benchmarks are allocation-free

Filtering:

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
    operators::Vector{Symbol}=FIRST_ORDER_OPERATORS,
    scenarios::Vector{<:Scenario}=default_scenarios();
    correctness::Bool=true,
    type_stability::Bool=true,
    call_count::Bool=false,
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
    benchmark_data = nothing
    set = @testset verbose = true "Backend tests" begin
        if correctness
            test_correctness(backends, operators, scenarios)
        end
        if type_stability
            test_type_stability(backends, operators, scenarios)
        end
        if call_count
            test_call_count(backends, operators, scenarios)
        end
        if benchmark || allocations
            benchmark_data = run_benchmark(
                backends, operators, scenarios; allocations=allocations
            )
        end
    end
    if benchmark
        return benchmark_data
    else
        return set
    end
end

"""
$(TYPEDSIGNATURES)

Shortcut for a single backend.
"""
function test_operators(backend::AbstractADType, args...; kwargs...)
    return test_operators([backend], args...; kwargs...)
end
