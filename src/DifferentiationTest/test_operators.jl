function test_correctness end
function test_type_stability end
function run_benchmark! end

const FIRST_ORDER_OPERATORS = [
    value_and_pushforward,
    value_and_pullback,
    value_and_derivative,
    value_and_gradient,
    value_and_jacobian,
]

const ALL_OPERATORS = vcat(FIRST_ORDER_OPERATORS)

function filter_operators(
    operators::Vector{<:Function};
    first_order::Bool,
    second_order::Bool,
    excluded::Vector{<:Function},
)
    !first_order && (operators = filter(!in(FIRST_ORDER_OPERATORS), operators))
    !second_order && (operators = filter(in(FIRST_ORDER_OPERATORS), operators))
    operators = filter(!in(excluded), operators)
    return operators
end

function filter_scenarios(
    scenarios::Vector{Scenario};
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

"""
    test_operators(backends, [operators, scenarios]; [kwargs...])

Cross-test a list of `backends` for a list of `operators` on a list of `scenarios`, running a variety of different tests.

- If `benchmark` is `false`, this returns a `TestSet` object.
- If `benchmark` is `true`, this returns a [`BenchmarkData`](@ref) object, which is easy to turn into a `DataFrame`.

# Default arguments

- `operators`: `$FIRST_ORDER_OPERATORS`
- `scenarios`: the output of [`default_scenarios()`](@ref)

# Keyword arguments

Testing:

- `correctness=true`: whether to compare the differentiation results with those given by ForwardDiff.jl
- `type_stability=true`: whether to check type stability with JET.jl
- `call_count=false`: whether to check that the function is called the right number of times
- `benchmark=false`: whether to run and return a benchmark suite with Chairmarks.jl
- `allocations=false`: whether to check that the benchmarks are allocation-free
- `detailed=false`: whether to print a detailed test set (by scenario) or condensed test set (by operator)

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
    operators::Vector{<:Function}=ALL_OPERATORS,
    scenarios::Vector{<:Scenario}=default_scenarios();
    # testing
    correctness::Bool=true,
    type_stability::Bool=true,
    call_count::Bool=false,
    benchmark::Bool=false,
    allocations::Bool=false,
    detailed=false,
    # filtering
    input_type::Type=Any,
    output_type::Type=Any,
    first_order=true,
    second_order=true,
    allocating=true,
    mutating=true,
    excluded::Vector{<:Function}=Function[],
)
    operators = filter_operators(operators; first_order, second_order, excluded)
    scenarios = filter_scenarios(scenarios; input_type, output_type, allocating, mutating)

    benchmark_data = BenchmarkData()

    title =
        "Differentiation tests -" *
        (correctness ? " correctness" : "") *
        (type_stability ? " types" : "") *
        (call_count ? " calls" : "") *
        (benchmark ? " benchmark" : "") *
        (allocations ? " allocations" : "")
    test_set = @testset verbose = true "$title" begin
        @testset verbose = true "$(backend_string(backend))" for backend in backends
            @testset verbose = detailed "$op" for op in operators
                @testset "$scen" for scen in filter(scenarios) do scen
                    compatible(backend, op, scen)
                end
                    if correctness
                        @testset verbose = true "Call count" begin
                            test_correctness(backend, op, scen)
                        end
                    end
                    if type_stability
                        @testset verbose = true "Type stability" begin
                            test_type_stability(backend, op, scen)
                        end
                    end
                    if call_count
                        @testset verbose = true "Call count" begin
                            test_call_count(backend, op, scen)
                        end
                    end
                    if benchmark || allocations
                        @testset verbose = true "Allocations" begin
                            run_benchmark!(
                                benchmark_data, backend, op, scen; allocations=allocations
                            )
                        end
                    end
                end
            end
        end
    end

    if benchmark
        return benchmark_data
    else
        return test_set
    end
end

"""
$(TYPEDSIGNATURES)

Shortcut for a single backend.
"""
function test_operators(backend::AbstractADType, args...; kwargs...)
    return test_operators([backend], args...; kwargs...)
end
