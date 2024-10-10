using ADTypes
using DifferentiationInterface
using DifferentiationInterface: AutoZeroForward, AutoZeroReverse
using DifferentiationInterfaceTest
using DifferentiationInterfaceTest: allocfree_scenarios

using Test

LOGGING = get(ENV, "CI", "false") == "false"

## Type stability

test_differentiation(
    AutoZeroForward(),
    default_scenarios(; include_batchified=false);
    correctness=false,
    type_stability=:full,
    logging=LOGGING,
)

test_differentiation(
    AutoZeroReverse(),
    default_scenarios(; include_batchified=false);
    correctness=false,
    type_stability=:prepared,
    logging=LOGGING,
)

## Benchmark

data0 = benchmark_differentiation(
    AutoZeroForward(),
    default_scenarios(; include_batchified=false, include_constantified=true);
    logging=LOGGING,
);

data1 = benchmark_differentiation(
    AutoZeroForward(),
    default_scenarios(; include_batchified=false);
    benchmark=:full,
    logging=LOGGING,
);

struct FakeBackend <: ADTypes.AbstractADType end
ADTypes.mode(::FakeBackend) = ADTypes.ForwardMode()

data2 = benchmark_differentiation(
    FakeBackend(),
    default_scenarios(; include_batchified=false);
    count_calls=false,
    logging=false,
);

@testset "Benchmarking DataFrame" begin
    for col in eachcol(data1)
        if eltype(col) <: AbstractFloat
            @test !any(isnan, col)
        end
    end
    for col in eachcol(data2)
        if eltype(col) <: AbstractFloat
            @test all(isnan, col)
        end
    end
end

## Allocations

data_allocfree = vcat(
    benchmark_differentiation(
        AutoZeroForward(),
        allocfree_scenarios();
        excluded=[:pullback, :gradient],
        benchmark=:prepared,
        logging=LOGGING,
    ),
    benchmark_differentiation(
        AutoZeroReverse(),
        allocfree_scenarios();
        excluded=[:pushforward, :derivative],
        benchmark=:prepared,
        logging=LOGGING,
    ),
)

@test all(iszero, data_allocfree[!, :allocs])
