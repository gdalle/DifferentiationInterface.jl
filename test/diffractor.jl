using ADTypes: AutoDiffractor
using Diffractor: Diffractor
using DifferentiationInterface.DifferentiationTest

test_operators(AutoDiffractor(); second_order=false, type_stability=false);
