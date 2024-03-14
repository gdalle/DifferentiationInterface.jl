using ADTypes: AutoDiffractor
using Diffractor: Diffractor
using DifferentiationInterface.DifferentiationTest

test_all_operators(AutoDiffractor(), default_scenarios(); type_stability=false);
