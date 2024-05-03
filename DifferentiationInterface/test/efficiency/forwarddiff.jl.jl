using DataFrames: DataFrame
using DifferentiationInterface, DifferentiationInterfaceTest
import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using Symbolics: Symbolics
using Test

@testset verbose = false "Dense" begin
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

@testset verbose = false "Sparse" begin
    # sparse jacobian for f!(x, y)

    b_sparse = AutoSparse(
        AutoForwardDiff(; chunksize=1);
        sparsity_detector=DI.SymbolicsSparsityDetector(),
        coloring_algorithm=DI.GreedyColoringAlgorithm(),
    )

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
