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

test_operators([AutoZeroForward(), AutoZeroReverse()]; correctness=false);

test_operators(
    [
        SecondOrder(AutoZeroForward(), AutoZeroReverse()),
        SecondOrder(AutoZeroReverse(), AutoZeroForward()),
    ];
    first_order=false,
    correctness=false,
);

# allocs (experimental)

result = test_operators(
    [AutoZeroForward(), AutoZeroReverse()];
    correctness=false,
    type_stability=false,
    benchmark=true,
    allocations=true,
);

data = parse_benchmark(result)
