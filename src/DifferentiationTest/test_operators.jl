test_correctness(args...; kwargs...) = error("Please load ForwardDiff.jl")
test_type_stability(args...; kwargs...) = error("Please load JET.jl")

const FIRST_ORDER_OPERATORS = [
    PushforwardAllocating(),
    PushforwardMutating(),
    PullbackAllocating(),
    PullbackMutating(),
    MultiderivativeAllocating(),
    MultiderivativeMutating(),
    DerivativeAllocating(),
    # DerivativeMutating(),
    GradientAllocating(),
    # GradientMutating(),
    JacobianAllocating(),
    JacobianMutating(),
]

const SECOND_ORDER_OPERATORS = [
    SecondDerivativeAllocating(),
    # SecondDerivativeMutating(),
    HessianAllocating(),
    # HessianMutating(),
    HessianVectorProductAllocating(),
    # HessianVectorProductMutating(),
]

const ALL_OPERATORS = vcat(FIRST_ORDER_OPERATORS, SECOND_ORDER_OPERATORS)

function filter_operators(
    operators::Vector{<:AbstractOperator};
    first_order::Bool,
    second_order::Bool,
    allocating::Bool,
    mutating::Bool,
    excluded::Vector{<:AbstractOperator},
)
    !first_order && (operators = filter(!isfirstorder, operators))
    !second_order && (operators = filter(!issecondorder, operators))
    !allocating && (operators = filter(!isallocating, operators))
    !mutating && (operators = filter(!ismutating, operators))
    operators = filter(op -> !in(op, excluded), operators)
    return operators
end

"""
    test_operators(backends, [operators, scenarios]; [kwargs...])

Cross-test a list of `backends` for a list of `operators` on a list of `scenarios.`

Return `nothing`, except when `benchmark=true`.

# Default arguments

- `operators`: defaults to all operators
- `scenarios`: defaults to a set of default scenarios

# Keyword arguments

- `correctness=true`: whether to compare the differentiation results with those given by ForwardDiff.jl
- `type_stability=true`: whether to check type stability with JET.jl
- `call_count=false`: whether to check that the function is called the right number of times
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
    operators::Vector{<:AbstractOperator}=ALL_OPERATORS,
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
    excluded::Vector{<:AbstractOperator}=AbstractOperator[],
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
