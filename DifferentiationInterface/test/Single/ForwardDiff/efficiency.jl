using DataFrames: DataFrame
using DifferentiationInterface, DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using SparseConnectivityTracer
using SparseMatrixColorings
using Test

function vecexp!(y, x)
    y .= exp.(x)
    return nothing
end

sumexp(x) = sum(exp, x)

@testset verbose = false "Dense efficiency" begin
    # derivative and gradient for `f(x)`

    results1 = benchmark_differentiation(
        [AutoForwardDiff()],
        [
            DerivativeScenario(exp; x=1.0, place=:outofplace),
            GradientScenario(sumexp; x=rand(10), place=:inplace),
        ];
        logging=LOGGING,
    )

    # derivative and jacobian for f!(x, y)

    results2 = benchmark_differentiation(
        [AutoForwardDiff()],
        [
            DerivativeScenario(vecexp!; x=1.0, y=zeros(10), place=:inplace),
            JacobianScenario(vecexp!; x=rand(10), y=zeros(10), place=:inplace),
        ];
        logging=LOGGING,
    )

    data = vcat(DataFrame(results1), DataFrame(results2))

    useless_rows =
        startswith.(string.(data[!, :operator]), Ref("prepare")) .|
        startswith.(string.(data[!, :operator]), Ref("value_and"))

    useful_data = data[.!useless_rows, :]

    for row in eachrow(useful_data)
        scen = row[:scenario]
        @testset "$(row[:operator]) - $(string(scen.f)) : $(typeof(scen.x)) -> $(typeof(scen.y))" begin
            VERSION >= v"1.10" && @test row[:allocs] == 0
        end
    end
end

@testset verbose = false "Sparse efficiency" begin
    # sparse jacobian for f!(x, y)

    results1 = benchmark_differentiation(
        [
            AutoSparse(
                AutoForwardDiff(; chunksize=1);
                sparsity_detector=TracerSparsityDetector(),
                coloring_algorithm=GreedyColoringAlgorithm(),
            ),
        ],
        sparse_scenarios();
        input_type=AbstractVector,
        output_type=AbstractVector,
        outofplace=false,
        onearg=false,
        second_order=false,
        logging=LOGGING,
    )

    data = vcat(DataFrame(results1))

    useless_rows = startswith.(string.(data[!, :operator]), Ref("prepare"))

    useful_data = data[.!useless_rows, :]

    for row in eachrow(useful_data)
        scen = row[:scenario]
        @testset "$(row[:operator]) - $(string(scen.f)) : $(typeof(scen.x)) -> $(typeof(scen.y))" begin
            VERSION >= v"1.10" && @test row[:allocs] == 0
            @test row[:calls] < prod(size(scen.x))
        end
    end
end
