using DifferentiationInterface
using DifferentiationInterfaceTest
using DifferentiationInterfaceTest: AutoZeroForward, AutoZeroReverse, scenario_to_zero

using DataFrames: DataFrames
using Test

@test check_available(AutoZeroForward())
@test check_available(AutoZeroReverse())
@test check_twoarg(AutoZeroForward())
@test check_twoarg(AutoZeroReverse())

## Correctness + type stability

test_differentiation(
    [AutoZeroForward(), AutoZeroReverse()],
    scenario_to_zero.(default_scenarios());
    correctness=true,
    type_stability=true,
    logging=LOGGING,
)

test_differentiation(
    [
        SecondOrder(AutoZeroForward(), AutoZeroReverse()),
        SecondOrder(AutoZeroReverse(), AutoZeroForward()),
    ],
    scenario_to_zero.(default_scenarios());
    correctness=true,
    type_stability=true,
    first_order=false,
    logging=LOGGING,
)

## Benchmark

data1 = benchmark_differentiation(
    [AutoZeroForward(), AutoZeroReverse()], default_scenarios(); logging=LOGGING
);

data2 = benchmark_differentiation(
    [SecondOrder(AutoZeroForward(), AutoZeroReverse())],
    default_scenarios();
    first_order=false,
    logging=LOGGING,
);

struct FakeBackend <: ADTypes.AbstractADType end
ADTypes.mode(::FakeBackend) = ADTypes.ForwardMode()

data3 = benchmark_differentiation([FakeBackend()], default_scenarios(); logging=false);

@testset "Benchmarking DataFrame" begin
    for col in eachcol(vcat(data1, data2))
        if eltype(col) <: AbstractFloat
            @test !any(isnan, col)
        end
    end
    for col in eachcol(data3)
        if eltype(col) <: AbstractFloat
            @test all(isnan, col)
        end
    end
end

## Weird arrays

test_differentiation(
    [AutoZeroForward(), AutoZeroReverse()],
    scenario_to_zero.(vcat(component_scenarios(), gpu_scenarios(), static_scenarios()));
    correctness=true,
    logging=LOGGING,
)
