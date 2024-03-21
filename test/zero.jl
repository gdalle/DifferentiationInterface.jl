using DifferentiationInterface
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

# call count (experimental)

test_operators(
    AutoZeroForward();
    correctness=false,
    type_stability=false,
    call_count=true,
    second_order=false,
);

test_operators(
    AutoZeroReverse();
    correctness=false,
    type_stability=false,
    call_count=true,
    second_order=false,
    excluded=[:multiderivative_allocating],
);

# allocs (experimental)

test_operators(
    [AutoZeroForward(), AutoZeroReverse()];
    correctness=false,
    type_stability=false,
    benchmark=true,
    allocations=true,
    second_order=false,
);

data = test_operators(
    [AutoZeroForward(), AutoZeroReverse()];
    correctness=false,
    type_stability=false,
    benchmark=true,
);

df = DataFrames.DataFrame(pairs(data)...)
