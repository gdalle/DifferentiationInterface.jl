using ADTypes: AutoDiffractor
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using Diffractor: Diffractor

using ForwardDiff: ForwardDiff
using JET: JET
using Test

@test check_available(AutoDiffractor())
@test !check_mutation(AutoDiffractor())
@test_broken check_hessian(AutoDiffractor())

test_operators(AutoDiffractor(); second_order=false, type_stability=false);
