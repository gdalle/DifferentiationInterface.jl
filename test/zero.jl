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

# call count (experimental)

test_operators(
    [AutoZeroForward(), AutoZeroReverse()],
    [
        :pushforward_allocating,
        :pushforward_mutating,
        :pullback_allocating,
        :pullback_mutating,
        :derivative_allocating,
        :multiderivative_allocating,
        :multiderivative_mutating,
        :gradient_allocating,
        :jacobian_allocating,
        :jacobian_mutating,
        :second_derivative_allocating,
        :hessian_vector_product_allocating,
        :hessian_allocating,
    ];
    correctness=false,
    type_stability=false,
    call_count=true,
);
