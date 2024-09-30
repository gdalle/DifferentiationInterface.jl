using ADTypes
using DifferentiationInterface
using DifferentiationInterface: AutoZeroForward, AutoZeroReverse
using DifferentiationInterfaceTest
using DifferentiationInterfaceTest: test_allocfree, allocfree_scenarios

using Test

LOGGING = get(ENV, "CI", "false") == "false"

## Type stability

test_differentiation(
    [AutoZeroForward(), AutoZeroReverse()],
    zero.(default_scenarios());
    correctness=true,
    type_stability=true,
    logging=LOGGING,
)

## Benchmark

data1 = benchmark_differentiation(
    [AutoZeroForward()], default_scenarios(; include_constantified=true); logging=LOGGING
);

struct FakeBackend <: ADTypes.AbstractADType end
ADTypes.mode(::FakeBackend) = ADTypes.ForwardMode()

data2 = benchmark_differentiation([FakeBackend()], default_scenarios(); logging=false);

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
        [AutoZeroForward()],
        allocfree_scenarios();
        excluded=[:pullback, :gradient],
        logging=LOGGING,
    ),
    benchmark_differentiation(
        [AutoZeroReverse()],
        allocfree_scenarios();
        excluded=[:pushforward, :derivative],
        logging=LOGGING,
    ),
)

test_allocfree(data_allocfree);
