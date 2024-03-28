function filter_scenarios(
    scenarios::Vector{<:AbstractScenario};
    first_order::Bool,
    second_order::Bool,
    input_type::Type,
    output_type::Type,
    allocating::Bool,
    mutating::Bool,
)
    !first_order && (scenarios = filter(s -> isa(s, AbstractFirstOrderScenario), scenarios))
    !second_order &&
        (scenarios = filter(s -> isa(s, AbstractSecondOrderScenario), scenarios))
    scenarios = filter(scenarios) do scen
        typeof(scen.x) <: input_type && typeof(scen.y) <: output_type
    end
    !allocating && (scenarios = filter(ismutating, scenarios))
    !mutating && (scenarios = filter(!ismutating, scenarios))
    return scenarios
end

"""
    test_differentiation(backends, [operators, scenarios]; [kwargs...])

Cross-test a list of `backends` for a list of `operators` on a list of `scenarios`, running a variety of different tests.

If `benchmark=true`, return a [`BenchmarkData`](@ref) object, otherwise return `nothing`.

# Default arguments

- `scenarios::Vector{Scenario}`: the output of [`default_scenarios()`](@ref)

# Keyword arguments

Testing:

- `correctness=true`: whether to compare the differentiation results with the theoretical values specified in each scenario. If a backend object like `correctness=AutoForwardDiff()` is passed instead of a boolean, the results will be compared using that reference backend as the ground truth. 
- `call_count=false`: whether to check that the function is called the right number of times
- `type_stability=false`: whether to check type stability with JET.jl (thanks to `@test_opt`)
- `benchmark=false`: whether to run and return a benchmark suite with Chairmarks.jl
- `allocations=false`: whether to check that the benchmarks are allocation-free
- `detailed=false`: whether to print a detailed test set (by scenario) or condensed test set (by operator)

Filtering:

- `input_type=Any`: restrict scenario inputs to subtypes of this
- `output_type=Any`: restrict scenario outputs to subtypes of this
- `allocating=true`: consider operators for allocating functions
- `mutating=true`: consider operators for mutating functions
- `first_order=true`: consider first order operators
- `second_order=true`: consider second order operators
- `excluded=Symbol[]`: list of excluded operators

Options:

- `isapprox=isapprox`: function used to compare objects, only needs to be set for complicated cases beyond arrays / scalars
- `rtol=1e-3`: precision for correctness testing (when comparing to the reference outputs)
"""
function test_differentiation(
    backends::Vector{<:AbstractADType},
    scenarios::Vector{<:AbstractScenario}=default_scenarios();
    # testing
    correctness::Union{Bool,AbstractADType}=true,
    type_stability::Bool=false,
    call_count::Bool=false,
    benchmark::Bool=false,
    allocations::Bool=false,
    detailed=false,
    # filtering
    input_type::Type=Any,
    output_type::Type=Any,
    allocating=true,
    mutating=true,
    first_order=true,
    second_order=true,
    excluded::Vector{<:Function}=Function[],
    # options
    isapprox=isapprox,
    rtol=1e-3,
)
    scenarios = filter_scenarios(
        scenarios; first_order, second_order, input_type, output_type, allocating, mutating
    )

    benchmark_data = BenchmarkData()

    title =
        "Differentiation tests -" *
        (correctness != false ? " correctness" : "") *
        (call_count ? " calls" : "") *
        (type_stability ? " types" : "") *
        (benchmark ? " benchmark" : "") *
        (allocations ? " allocations" : "")

    @testset verbose = detailed "$(backend_string(backend))" for backend in backends
        @testset "$scen" for scen in filter(s -> compatible(backend, s), scenarios)
            if correctness != false
                @testset "Correctness" begin
                    if correctness isa AbstractADType
                        test_correctness(
                            backend, change_ref(scen, correctness); isapprox, rtol
                        )
                    else
                        test_correctness(backend, scen; isapprox, rtol)
                    end
                end
            end
            if call_count
                @testset "Call count" begin
                    test_call_count(backend, scen)
                end
            end
            if type_stability
                @testset "Type stability" begin
                    test_jet(backend, scen)
                end
            end
            if benchmark || allocations
                @testset "Allocations" begin
                    run_benchmark!(benchmark_data, backend, scen; allocations=allocations)
                end
            end
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
function test_differentiation(backend::AbstractADType, args...; kwargs...)
    return test_differentiation([backend], args...; kwargs...)
end
