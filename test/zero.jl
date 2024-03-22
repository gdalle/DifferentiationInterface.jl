using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using DifferentiationInterface.DifferentiationTest: AutoZeroForward, AutoZeroReverse

using Chairmarks: Chairmarks
using DataFrames: DataFrames
using JET: JET
using Test

@test check_available(AutoZeroForward())
@test check_available(AutoZeroReverse())

test_operators(
    [AutoZeroForward(), AutoZeroReverse()]; second_order=false, correctness=false
);

# call count (experimental)

test_operators(
    AutoZeroForward();
    correctness=false,
    type_stability=false,
    call_count=true,
    second_order=false,
    excluded=[:gradient_allocating],
);

test_operators(
    AutoZeroReverse();
    correctness=false,
    type_stability=false,
    call_count=true,
    second_order=true,
    excluded=[:derivative_allocating],
);

# allocs (experimental)

test_operators(
    [AutoZeroForward(), AutoZeroReverse()];
    correctness=false,
    type_stability=false,
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
