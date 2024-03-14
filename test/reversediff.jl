using ADTypes: AutoReverseDiff
using ReverseDiff: ReverseDiff
using DifferentiationInterface.DifferentiationTest

test_all_operators(AutoReverseDiff(), default_scenarios(); type_stability=false);
