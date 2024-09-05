"""
$(TYPEDSIGNATURES)

Cross-test a list of `backends` on a list of `scenarios`, running a variety of different tests.

# Default arguments

- `scenarios::Vector{<:Scenario}`: the output of [`default_scenarios()`](@ref)

# Keyword arguments

Testing:

- `correctness=true`: whether to compare the differentiation results with the theoretical values specified in each scenario
- `type_stability=false`: whether to check type stability with JET.jl (thanks to `JET.@test_opt`)
- `sparsity`: whether to check sparsity of the jacobian / hessian
- `detailed=false`: whether to print a detailed or condensed test log

Filtering:

- `input_type=Any`, `output_type=Any`: restrict scenario inputs / outputs to subtypes of this
- `first_order=true`, `second_order=true`: include first order / second order operators
- `onearg=true`, `twoarg=true`: include out-of-place / in-place functions
- `inplace=true`, `outofplace=true`: include in-place / out-of-place operators

Options:

- `logging=false`: whether to log progress
- `isequal=isequal`: function used to compare objects exactly, with the standard signature `isequal(x, y)`
- `isapprox=isapprox`: function used to compare objects approximately, with the standard signature `isapprox(x, y; atol, rtol)`
- `atol=0`: absolute precision for correctness testing (when comparing to the reference outputs)
- `rtol=1e-3`: relative precision for correctness testing (when comparing to the reference outputs)
"""
function test_differentiation(
    backends::Vector{<:AbstractADType},
    scenarios::Vector{<:Scenario}=default_scenarios();
    # testing
    correctness::Bool=true,
    type_stability::Bool=false,
    call_count::Bool=false,
    sparsity::Bool=false,
    detailed=false,
    # filtering
    input_type::Type=Any,
    output_type::Type=Any,
    first_order::Bool=true,
    second_order::Bool=true,
    onearg::Bool=true,
    twoarg::Bool=true,
    inplace::Bool=true,
    outofplace::Bool=true,
    excluded::Vector{Symbol}=Symbol[],
    # options
    logging::Bool=false,
    isequal=isequal,
    isapprox=isapprox,
    atol::Real=0,
    rtol::Real=1e-3,
)
    scenarios = filter_scenarios(
        scenarios;
        first_order,
        second_order,
        input_type,
        output_type,
        onearg,
        twoarg,
        inplace,
        outofplace,
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
        @testset verbose = detailed "$backend" for (i, backend) in enumerate(backends)
            filtered_scenarios = filter(s -> compatible(backend, s), scenarios)
            grouped_scenarios = group_by_operator(filtered_scenarios)
            @testset verbose = detailed "$op" for (j, (op, op_group)) in
                                                  enumerate(pairs(grouped_scenarios))
                @testset "$scen" for (k, scen) in enumerate(op_group)
                    next!(
                        prog;
                        showvalues=[
                            (:backend, "$backend - $i/$(length(backends))"),
                            (:scenario_type, "$op - $j/$(length(grouped_scenarios))"),
                            (:scenario, "$k/$(length(op_group))"),
                            (:arguments, nb_args(scen)),
                            (:place, place(scen)),
                            (:function, scen.f),
                            (:input_type, typeof(scen.x)),
                            (:input_size, mysize(scen.x)),
                            (:output_type, typeof(scen.y)),
                            (:output_size, mysize(scen.y)),
                            (:batched_seed, scen.seed isa Tangents),
                        ],
                    )
                    correctness && @testset "Correctness" begin
                        test_correctness(backend, scen; isequal, isapprox, atol, rtol)
                    end
                    type_stability && @testset "Type stability" begin
                        @static if VERSION >= v"1.7"
                            test_jet(backend, scen)
                        end
                    end
                    sparsity && @testset "Sparsity" begin
                        test_sparsity(backend, scen)
                    end
                    yield()
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

The object returned is a `DataFrames.DataFrame` where each column corresponds to a field of [`DifferentiationBenchmarkDataRow`](@ref).

The keyword arguments available here have the same meaning as those in [`test_differentiation`](@ref).
"""
function benchmark_differentiation(
    backends::Vector{<:AbstractADType},
    scenarios::Vector{<:Scenario};
    # filtering
    input_type::Type=Any,
    output_type::Type=Any,
    first_order::Bool=true,
    second_order::Bool=true,
    onearg::Bool=true,
    twoarg::Bool=true,
    inplace::Bool=true,
    outofplace::Bool=true,
    excluded::Vector{Symbol}=Symbol[],
    # options
    logging::Bool=false,
)
    scenarios = filter_scenarios(
        scenarios;
        first_order,
        second_order,
        input_type,
        output_type,
        onearg,
        twoarg,
        inplace,
        outofplace,
        excluded,
    )

    benchmark_data = DifferentiationBenchmarkDataRow[]
    prog = ProgressUnknown(; desc="Benchmarking", spinner=true, enabled=logging)
    for (i, backend) in enumerate(backends)
        filtered_scenarios = filter(s -> compatible(backend, s), scenarios)
        grouped_scenarios = group_by_operator(filtered_scenarios)
        for (j, (op, op_group)) in enumerate(pairs(grouped_scenarios))
            for (k, scen) in enumerate(op_group)
                next!(
                    prog;
                    showvalues=[
                        (:backend, "$backend - $i/$(length(backends))"),
                        (:scenario_type, "$op - $j/$(length(grouped_scenarios))"),
                        (:scenario, "$k/$(length(op_group))"),
                        (:arguments, nb_args(scen)),
                        (:place, place(scen)),
                        (:function, scen.f),
                        (:input_type, typeof(scen.x)),
                        (:input_size, mysize(scen.x)),
                        (:output_type, typeof(scen.y)),
                        (:output_size, mysize(scen.y)),
                        (:batched_seed, scen.seed isa Tangents),
                    ],
                )
                run_benchmark!(benchmark_data, backend, scen; logging)
                yield()
            end
        end
    end
    return DataFrame(benchmark_data)
end

"""
    test_allocfree(benchmark_data::DataFrame)

Test that every row in `benchmark_data` which is not a preparation row has zero allocation.
"""
function test_allocfree(benchmark_data::DataFrame)
    preparation_rows = startswith.(string.(benchmark_data[!, :operator]), Ref("prepare"))
    useful_data = benchmark_data[.!preparation_rows, :]

    @testset verbose = true "No allocations" begin
        @testset "$(row[:scenario]) - $(row[:operator])" for row in eachrow(useful_data)
            @test row[:allocs] == 0
        end
    end
end
