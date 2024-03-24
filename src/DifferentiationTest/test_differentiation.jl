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

function test_correctness(backend::AbstractADType, op::Function, scen::Scenario)
    return error("Please load ForwardDiff.jl or check your method signature")
end

function test_jet(backend::AbstractADType, op::Function, scen::Scenario; call, opt)
    return error("Please load JET.jl or check your method signature")
end

function run_benchmark!(
    data::BenchmarkData, backend::AbstractADType, op::Function, scen::Scenario; allocations
)
    return error("Please load Chairmarks.jl or check your method signature")
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
- `error_free=false`: whether to run it once and see if it errors
- `type_stability=false`: whether to check type stability with JET.jl (`@test_opt`)
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
function test_differentiation(
    backends::Vector{<:AbstractADType},
    operators::Vector{<:Function}=all_operators(),
    scenarios::Vector{<:Scenario}=default_scenarios();
    # testing
    correctness::Bool=true,
    error_free::Bool=false,
    type_stability::Bool=false,
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
        (error_free ? " errors" : "") *
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
                        @testset verbose = true "Correctness" begin
                            test_correctness(backend, op, scen)
                        end
                    end
                    if error_free
                        @testset verbose = true "Error-free" begin
                            test_jet(backend, op, scen; call=true, opt=false)
                        end
                    end
                    if type_stability
                        @testset verbose = true "Type stability" begin
                            test_jet(backend, op, scen; call=false, opt=true)
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
function test_differentiation(backend::AbstractADType, args...; kwargs...)
    return test_differentiation([backend], args...; kwargs...)
end
