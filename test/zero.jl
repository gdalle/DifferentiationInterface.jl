using DifferentiationInterface: AutoZeroForward, AutoZeroReverse
using DifferentiationInterface.DifferentiationTest

test_operators_allocating(AutoZeroForward(); correctness=false);
test_operators_mutating(AutoZeroForward(); correctness=false);

test_operators_allocating(AutoZeroReverse(); correctness=false);
test_operators_mutating(AutoZeroReverse(); correctness=false);

test_operators_allocating(
    SecondOrder(AutoZeroReverse(), AutoZeroForward());
    correctness=false,
    included=[:hessian],
)
