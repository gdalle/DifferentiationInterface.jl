using DifferentiationInterface.DifferentiationTest
using DifferentiationInterface.DifferentiationTest: AutoZeroForward, AutoZeroReverse

test_all_operators(
    AutoZeroForward(), default_scenarios(); correctness=false, type_stability=true
);

test_all_operators_mutating(
    AutoZeroForward(), default_scenarios(); correctness=false, type_stability=true
);

test_all_operators(
    AutoZeroReverse(), default_scenarios(); correctness=false, type_stability=true
);

test_all_operators_mutating(
    AutoZeroReverse(), default_scenarios(); correctness=false, type_stability=true
);
