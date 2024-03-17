using DifferentiationInterface: SecondOrder, AutoZeroForward, AutoZeroReverse
using DifferentiationInterface.DifferentiationTest

test_operators(AutoZeroForward(); correctness=false);
test_operators(AutoZeroReverse(); correctness=false);

test_operators(
    SecondOrder(AutoZeroForward(), AutoZeroReverse()); first_order=false, correctness=false
);
test_operators(
    SecondOrder(AutoZeroReverse(), AutoZeroForward()); first_order=false, correctness=false
);
