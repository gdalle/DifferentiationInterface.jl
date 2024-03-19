using ADTypes: AutoDiffractor
using Diffractor: Diffractor
using DifferentiationInterface.DifferentiationTest
using Test

@test available(AutoDiffractor())

test_operators(AutoDiffractor(); second_order=false, type_stability=false);
