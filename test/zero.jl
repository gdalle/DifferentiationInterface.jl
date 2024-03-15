using DifferentiationInterface.DifferentiationTest
using DifferentiationInterface.DifferentiationTest: AutoZeroForward, AutoZeroReverse

test_operators_allocating(AutoZeroForward(); correctness=false);
test_operators_mutating(AutoZeroForward(); correctness=false);

test_operators_allocating(AutoZeroReverse(); correctness=false);
test_operators_mutating(AutoZeroReverse(); correctness=false);
