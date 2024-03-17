using DifferentiationInterface: SecondOrder, AutoZeroForward, AutoZeroReverse
using DifferentiationInterface.DifferentiationTest

test_operators_allocating(AutoZeroForward(); correctness=false);
test_operators_mutating(AutoZeroForward(); correctness=false);

test_operators_allocating(AutoZeroReverse(); correctness=false);
test_operators_mutating(AutoZeroReverse(); correctness=false);

test_second_order_operators_allocating(
    SecondOrder(AutoZeroForward(), AutoZeroForward()); correctness=false
)

test_second_order_operators_allocating(
    SecondOrder(AutoZeroReverse(), AutoZeroForward()); correctness=false
)

test_second_order_operators_allocating(
    SecondOrder(AutoZeroForward(), AutoZeroReverse()); correctness=false
)

test_second_order_operators_allocating(
    SecondOrder(AutoZeroReverse(), AutoZeroReverse()); correctness=false
)
