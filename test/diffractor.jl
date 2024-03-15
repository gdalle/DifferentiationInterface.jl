using ADTypes: AutoDiffractor
using Diffractor: Diffractor
using DifferentiationInterface.DifferentiationTest

test_operators_allocating(AutoDiffractor(); type_stability=false);
