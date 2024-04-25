"""
$(TYPEDSIGNATURES)

Cross-test a list of `backends` on a list of `scenarios`, running a variety of different tests.

# Default arguments

- `scenarios::Vector{<:AbstractScenario}`: the output of [`default_scenarios()`](@ref)

# Keyword arguments

Testing:

- `correctness=true`: whether to compare the differentiation results with the theoretical values specified in each scenario
- `type_stability=false`: whether to check type stability with JET.jl (thanks to `@test_opt`)
- `sparsity`: whether to check sparsity of the jacobian / hessian
- `ref_backend`: if not `nothing`, an `ADTypes.AbstractADType` object to use instead of the scenario-specific reference to provide true values
- `detailed=false`: whether to print a detailed or condensed test log

Filtering:

- `input_type=Any`: restrict scenario inputs to subtypes of this
- `output_type=Any`: restrict scenario outputs to subtypes of this
- `first_order=true`: include first order operators
- `second_order=true`: include second order operators

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
    correctness::Bool=true,
    type_stability::Bool=false,
    call_count::Bool=false,
    sparsity::Bool=false,
    ref_backend=nothing,
    detailed=false,
    # filtering
    input_type::Type=Any,
    output_type::Type=Any,
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
        scenarios; first_order, second_order, input_type, output_type, excluded
    )

    title_additions =
        (correctness != false ? " + correctness" : "") *
        (call_count ? " + calls" : "") *
        (type_stability ? " + types" : "") *
        (sparsity ? " + sparsity" : "")
    title = "Testing" * title_additions[3:end]

    prog = ProgressUnknown(; desc="$title", spinner=true, enabled=logging)

    @testset verbose = true "$title" begin
        @testset verbose = detailed "$(backend_string(backend))" for (i, backend) in
                                                                     enumerate(backends)
            filtered_scenarios = filter(s -> compatible(backend, s), scenarios)
            @testset "$scen" for (j, scen) in enumerate(filtered_scenarios)
                next!(
                    prog;
                    showvalues=[
                        (:backend, "$(backend_string(backend)) - $i/$(length(backends))"),
                        (
                            :scenario,
                            "$(scen_type(scen)) - $j/$(length(filtered_scenarios))",
                        ),
                        (:arguments, nb_args(scen)),
                        (:operator, operator_place(scen)),
                        (:function, scen.f),
                        (:input, typeof(scen.x)),
                        (:output, typeof(scen.y)),
                    ],
                )
                correctness && @testset "Correctness" begin
                    test_correctness(backend, scen; isapprox, atol, rtol, ref_backend)
                end
                type_stability && @testset "Type stability" begin
                    test_jet(backend, scen; ref_backend)
                end
                sparsity && @testset "Sparsity" begin
                    test_sparsity(backend, scen; ref_backend)
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
    first_order=true,
    second_order=true,
    excluded=[],
    # options
    logging=false,
)
    scenarios = filter_scenarios(
        scenarios; first_order, second_order, input_type, output_type, excluded
    )

    benchmark_data = BenchmarkDataRow[]
    prog = ProgressUnknown(; desc="Benchmarking", spinner=true, enabled=logging)
    for (i, backend) in enumerate(backends)
        filtered_scenarios = filter(s -> compatible(backend, s), scenarios)
        for (j, scen) in enumerate(filtered_scenarios)
            next!(
                prog;
                showvalues=[
                    (:backend, "$(backend_string(backend)) - $i/$(length(backends))"),
                    (:scenario, "$(scen_type(scen)) - $j/$(length(filtered_scenarios))"),
                    (:arguments, nb_args(scen)),
                    (:operator, operator_place(scen)),
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
