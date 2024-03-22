using ADTypes: AutoFiniteDiff
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using FiniteDiff: FiniteDiff

using ForwardDiff: ForwardDiff
using JET: JET
using Test

@test check_available(AutoFiniteDiff())
@test check_mutation(AutoFiniteDiff())

test_operators(AutoFiniteDiff(); second_order=false, type_stability=false);
