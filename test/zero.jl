using DifferentiationInterface
using DifferentiationInterface: AutoZeroForward, AutoZeroReverse
using DifferentiationInterface.DifferentiationTest

using Chairmarks: Chairmarks
using DataFrames: DataFrames
using JET: JET
using Test

@test available(AutoZeroForward())
@test available(AutoZeroReverse())
@test available(SecondOrder(AutoZeroForward(), AutoZeroReverse()))

test_operators(
    [AutoZeroForward(), AutoZeroReverse()]; second_order=false, correctness=false
);

test_operators(
    [
        SecondOrder(AutoZeroForward(), AutoZeroForward()),
        SecondOrder(AutoZeroForward(), AutoZeroReverse()),
        SecondOrder(AutoZeroReverse(), AutoZeroForward()),
        SecondOrder(AutoZeroReverse(), AutoZeroReverse()),
    ];
    first_order=false,
    correctness=false,
);

# allocs (experimental)

test_operators(
    [AutoZeroForward(), AutoZeroReverse()];
    correctness=false,
    type_stability=false,
    allocations=true,
    second_order=false,
);

result = test_operators(
    [AutoZeroForward(), AutoZeroReverse()];
    correctness=false,
    type_stability=false,
    benchmark=true,
);

data = parse_benchmark(result)
