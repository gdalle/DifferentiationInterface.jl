using DifferentiationInterface, DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff

for backend in [AutoForwardDiff(), AutoSparse(AutoForwardDiff())]
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation([AutoForwardDiff(), AutoSparse(AutoForwardDiff())]; logging=LOGGING);

test_differentiation(
    MyAutoSparse(AutoForwardDiff()), sparse_scenarios(); sparsity=true, logging=LOGGING
);

test_differentiation(
    AutoForwardDiff(),
    # ForwardDiff access individual indices
    vcat(component_scenarios(), static_scenarios());
    # jacobian is super slow for some reason
    excluded=[JacobianScenario],
    second_order=false,
    logging=LOGGING,
);

## Efficiency

@testset verbose = false "Dense efficiency" begin
    # derivative and gradient for `f(x)`

    results1 = benchmark_differentiation(
        [AutoForwardDiff()];
        outofplace=false,
        twoarg=false,
        input_type=Union{Number,AbstractVector},
        output_type=Number,
        second_order=false,
        excluded=[PullbackScenario],
        logging=get(ENV, "CI", "false") == "false",
    )

    # derivative and jacobian for f!(x, y)

    results2 = benchmark_differentiation(
        [AutoForwardDiff()];
        outofplace=false,
        onearg=false,
        input_type=Union{Number,AbstractVector},
        output_type=AbstractVector,
        second_order=false,
        excluded=[PullbackScenario],
        logging=get(ENV, "CI", "false") == "false",
    )

    data = vcat(DataFrame(results1), DataFrame(results2))

    useless_rows =
        startswith.(string.(data[!, :operator]), Ref("prepare")) .|
        startswith.(string.(data[!, :operator]), Ref("value_and"))

    useful_data = data[.!useless_rows, :]

    @testset "$(row[:operator]) - $(row[:func]) : $(row[:input_type]) -> $(row[:output_type])" for row in
                                                                                                   eachrow(
        useful_data
    )
        @test row[:allocs] == 0
    end
end;

@testset verbose = false "Sparse efficiency" begin
    # sparse jacobian for f!(x, y)

    b_sparse = MyAutoSparse(AutoForwardDiff(; chunksize=1);)

    results1 = benchmark_differentiation(
        [b_sparse],
        sparse_scenarios();
        input_type=AbstractVector,
        output_type=AbstractVector,
        outofplace=false,
        onearg=false,
        second_order=false,
        logging=get(ENV, "CI", "false") == "false",
    )

    data = vcat(DataFrame(results1))

    useless_rows = startswith.(string.(data[!, :operator]), Ref("prepare"))

    useful_data = data[.!useless_rows, :]

    @testset "$(row[:operator]) - $(row[:func]) : $(row[:input_type]) -> $(row[:output_type])" for row in
                                                                                                   eachrow(
        useful_data
    )
        @test row[:allocs] == 0
        @test row[:calls] < prod(row[:input_size])
    end
end;
