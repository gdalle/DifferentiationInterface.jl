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

"""
    test_differentiation(backends, [operators, scenarios]; [kwargs...])

Cross-test a list of `backends` for a list of `operators` on a list of `scenarios`, running a variety of different tests.

- If `benchmark` is `false`, this runs the tests and returns `nothing`.
- If `benchmark` is `true`, this runs the tests and returns a [`BenchmarkData`](@ref) object, which is easy to turn into a `DataFrame`.

# Default arguments

- `operators::Vector{Function}`: the list `[pushforward, pullback,derivative, gradient, jacobian, second_derivative, hvp, hessian]`
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

- `rtol=1e-3`: precision for correctness testing (when comparing to the reference outputs)
"""
function test_differentiation(
    backends::Vector{<:AbstractADType},
    operators::Vector{<:Function}=all_operators(),
    scenarios::Vector{<:Scenario}=default_scenarios();
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
    rtol=1e-3,
)
    operators = filter_operators(operators; first_order, second_order, excluded)
    scenarios = filter_scenarios(scenarios; input_type, output_type, allocating, mutating)

    benchmark_data = BenchmarkData()

    title =
        "Differentiation tests -" *
        (correctness != false ? " correctness" : "") *
        (call_count ? " calls" : "") *
        (type_stability ? " types" : "") *
        (benchmark ? " benchmark" : "") *
        (allocations ? " allocations" : "")

    jet_ext = if type_stability
        ext = get_extension(DifferentiationInterface, :DifferentiationInterfaceJETExt)
        @assert !isnothing(ext)
        ext
    else
        nothing
    end

    chairmarks_ext = if allocations || benchmark
        ext = get_extension(DifferentiationInterface, :DifferentiationInterfaceChairmarksExt)
        @assert !isnothing(ext)
        ext
    else
        nothing
    end

    @testset verbose = detailed "$(backend_string(backend))" for backend in backends
        @testset verbose = detailed "$op" for op in operators
            @testset "$scen" for scen in filter(scenarios) do scen
                compatible(backend, op, scen)
            end
                if correctness != false
                    @testset "Correctness" begin
                        if correctness isa AbstractADType
                            test_correctness(
                                backend, op, change_ref(scen, correctness); rtol
                            )
                        else
                            test_correctness(backend, op, scen; rtol)
                        end
                    end
                end
                if call_count
                    @testset "Call count" begin
                        test_call_count(backend, op, scen)
                    end
                end
                if type_stability
                    @testset "Type stability" begin
                        jet_ext.test_jet(backend, op, scen)
                    end
                end
                if benchmark || allocations
                    @testset "Allocations" begin
                        chairmarks_ext.run_benchmark!(
                            benchmark_data, backend, op, scen; allocations=allocations
                        )
                    end
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
