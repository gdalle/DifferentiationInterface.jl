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

- If `benchmark` is `false`, this returns a `TestSet` object.
- If `benchmark` is `true`, this returns a [`BenchmarkData`](@ref) object, which is easy to turn into a `DataFrame`.

# Default arguments

- `operators::Vector{Function}`: the list `[pushforward, pullback,derivative, gradient, jacobian, second_derivative, hvp, hessian]`
- `scenarios::Vector{Scenario}`: the output of [`default_scenarios()`](@ref)

# Keyword arguments

Testing:

- `correctness=true`: whether to compare the differentiation results with those given by ForwardDiff.jl
- `call_count=false`: whether to check that the function is called the right number of times
- `type_stability=false`: whether to check type stability with JET.jl (`@test_opt`)
- `benchmark=false`: whether to run and return a benchmark suite with Chairmarks.jl
- `allocations=false`: whether to check that the benchmarks are allocation-free
- `detailed=false`: whether to print a detailed test set (by scenario) or condensed test set (by operator)

Filtering:

- `input_type=Any`: restrict scenario inputs to subtypes of this
- `output_type=Any`: restrict scenario outputs to subtypes of this
- `allocating=true`: consider operators for allocating functions
- `mutating=true`: consider operators for mutating functions
- `first_order=true`: consider first order operators
- `second_order=false`: consider second order operators
- `excluded=Symbol[]`: list of excluded operators
"""
function test_differentiation(
    backends::Vector{<:AbstractADType},
    operators::Vector{<:Function}=all_operators(),
    scenarios::Vector{<:Scenario}=default_scenarios();
    # testing
    correctness::Bool=true,
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
    second_order=false,
    excluded::Vector{<:Function}=Function[],
)
    operators = filter_operators(operators; first_order, second_order, excluded)
    scenarios = filter_scenarios(scenarios; input_type, output_type, allocating, mutating)

    benchmark_data = BenchmarkData()

    title =
        "Differentiation tests -" *
        (correctness ? " correctness" : "") *
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

    test_set = @testset verbose = true "$title" begin
        @testset verbose = true "$(backend_string(backend))" for backend in backends
            @testset verbose = detailed "$op" for op in operators
                @testset "$scen" for scen in filter(scenarios) do scen
                    compatible(backend, op, scen)
                end
                    if correctness
                        @testset "Correctness" begin
                            test_correctness(backend, op, scen)
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
function test_differentiation(backend::AbstractADType, args...; kwargs...)
    return test_differentiation([backend], args...; kwargs...)
end
