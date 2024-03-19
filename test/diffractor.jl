using ADTypes: AutoDiffractor
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using Diffractor: Diffractor
using Test

@test available(AutoDiffractor())

test_operators(AutoDiffractor(); second_order=false, type_stability=false);
