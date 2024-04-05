"""
$(TYPEDSIGNATURES)

Cross-test a list of `backends` on a list of `scenarios`, running a variety of different tests.

# Default arguments

- `scenarios::Vector{<:AbstractScenario}`: the output of [`default_scenarios()`](@ref)

# Keyword arguments

Testing:

- `correctness=true`: whether to compare the differentiation results with the theoretical values specified in each scenario
    - If a backend object like `correctness=AutoForwardDiff()` is passed instead of a boolean, the results will be compared using that reference backend as the ground truth.
    - Otherwise, the scenario-specific reference operator will be used as the ground truth instead, see [`AbstractScenario`](@ref) for details.
- `call_count=false`: whether to check that the function is called the right number of times
- `type_stability=false`: whether to check type stability with JET.jl (thanks to `@test_opt`)
- `sparsity`: whether to check sparsity of the jacobian / hessian
- `detailed=false`: whether to print a detailed or condensed test log

Filtering:

- `input_type=Any`: restrict scenario inputs to subtypes of this
- `output_type=Any`: restrict scenario outputs to subtypes of this
- `allocating=true`: consider operators for allocating functions
- `mutating=true`: consider operators for mutating functions
- `first_order=true`: consider first order operators
- `second_order=true`: consider second order operators

Options:

- `logging=false`: whether to log progress
- `isapprox=isapprox`: function used to compare objects, with the standard signature `isapprox(x, y; atol, rtol)`
- `atol=0`: absolute precision for correctness testing (when comparing to the reference outputs)
- `rtol=1e-3`: relative precision for correctness testing (when comparing to the reference outputs)
"""
function test_differentiation(
    backends::Vector{<:AbstractADType},
    scenarios::Vector{<:AbstractScenario}=default_scenarios();
    # testing
    correctness::Union{Bool,AbstractADType}=true,
    type_stability::Bool=false,
    call_count::Bool=false,
    sparsity::Bool=false,
    detailed=false,
    # filtering
    input_type::Type=Any,
    output_type::Type=Any,
    allocating=true,
    mutating=true,
    first_order=true,
    second_order=true,
    excluded=[],
    # options
    logging=false,
    isapprox=isapprox,
    atol=0,
    rtol=1e-3,
)
    scenarios = filter_scenarios(
        scenarios;
        first_order,
        second_order,
        input_type,
        output_type,
        allocating,
        mutating,
        excluded,
    )

    title_additions =
        (correctness != false ? " + correctness" : "") *
        (call_count ? " + calls" : "") *
        (type_stability ? " + types" : "") *
        (sparsity ? " + sparsity" : "")
    title = "Testing" * title_additions[3:end]

    prog = ProgressUnknown(; desc="$title", spinner=true, enabled=logging)

    @testset verbose = true "$title" begin
        @testset verbose = detailed "$(backend_string(backend))" for backend in backends
            @testset "$scen" for scen in filter(s -> compatible(backend, s), scenarios)
                next!(
                    prog;
                    showvalues=[
                        (:backend, backend_string(backend)),
                        (:scenario, typeof(scen).name.name),
                        (:function, scen.f),
                        (:input, typeof(scen.x)),
                        (:output, typeof(scen.y)),
                    ],
                )
                correctness != false && @testset "Correctness" begin
                    if correctness isa AbstractADType
                        test_correctness(
                            backend, scen; isapprox, atol, rtol, ref_backend=correctness
                        )
                    else
                        test_correctness(
                            backend, scen; isapprox, atol, rtol, ref_backend=nothing
                        )
                    end
                end
                call_count && @testset "Call count" begin
                    test_call_count(backend, scen)
                end
                type_stability && @testset "Type stability" begin
                    test_jet(backend, scen)
                end
                sparsity && @testset "Sparsity" begin
                    test_sparsity(backend, scen; ref_backend=nothing)
                end
            end
        end
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Shortcut for a single backend.
"""
function test_differentiation(backend::AbstractADType, args...; kwargs...)
    return test_differentiation([backend], args...; kwargs...)
end

"""
$(TYPEDSIGNATURES)

Benchmark a list of `backends` for a list of `operators` on a list of `scenarios`.

# Keyword arguments

- filtering: same as [`test_differentiation`](@ref) for the filtering part.
- `logging=false`: whether to log progress
"""
function benchmark_differentiation(
    backends::Vector{<:AbstractADType},
    scenarios::Vector{<:AbstractScenario}=default_scenarios();
    # filtering
    input_type::Type=Any,
    output_type::Type=Any,
    allocating=true,
    mutating=true,
    first_order=true,
    second_order=true,
    excluded=[],
    # options
    logging=false,
)
    scenarios = filter_scenarios(
        scenarios;
        first_order,
        second_order,
        input_type,
        output_type,
        allocating,
        mutating,
        excluded,
    )

    benchmark_data = BenchmarkDataRow[]
    prog = ProgressUnknown(; desc="Benchmarking", spinner=true, enabled=logging)
    for backend in backends
        for scen in filter(s -> compatible(backend, s), scenarios)
            next!(
                prog;
                showvalues=[
                    (:backend, backend_string(backend)),
                    (:scenario, typeof(scen).name.name),
                    (:function, scen.f),
                    (:input, typeof(scen.x)),
                    (:output, typeof(scen.y)),
                ],
            )
            run_benchmark!(benchmark_data, backend, scen)
        end
    end
    return benchmark_data
end
