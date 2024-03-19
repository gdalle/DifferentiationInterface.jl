using Chairmarks: Chairmarks
using DataFrames: DataFrames
using DifferentiationInterface
using DifferentiationInterface: AutoZeroForward, AutoZeroReverse
using DifferentiationInterface.DifferentiationTest
using Test

@test available(AutoZeroForward())
@test available(AutoZeroReverse())
@test available(SecondOrder(AutoZeroForward(), AutoZeroReverse()))

test_operators(AutoZeroForward(); correctness=false);
test_operators(AutoZeroReverse(); correctness=false);

test_operators(
    SecondOrder(AutoZeroForward(), AutoZeroReverse()); first_order=false, correctness=false
);
test_operators(
    SecondOrder(AutoZeroReverse(), AutoZeroForward()); first_order=false, correctness=false
);

# allocs (experimental)

result = test_operators(
    AutoZeroForward();
    correctness=false,
    type_stability=false,
    benchmark=true,
    allocations=true,
);

result = test_operators(
    AutoZeroReverse();
    correctness=false,
    type_stability=false,
    benchmark=true,
    allocations=true,
);
