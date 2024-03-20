using ADTypes: AutoDiffractor
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using Diffractor: Diffractor

using ForwardDiff: ForwardDiff
using JET: JET
using Test

@test available(AutoDiffractor())
@test !supports_mutation(AutoDiffractor())

test_operators(AutoDiffractor(); second_order=false, type_stability=false);
