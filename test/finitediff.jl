using ADTypes: AutoFiniteDiff
using FiniteDiff: FiniteDiff
using DifferentiationInterface.DifferentiationTest

test_all_operators(AutoFiniteDiff(), default_scenarios(); type_stability=false);
