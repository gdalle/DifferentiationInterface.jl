using ADTypes: AutoForwardDiff
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using ForwardDiff: ForwardDiff

using JET: JET
using Test

@test check_available(AutoForwardDiff())
@test check_mutation(AutoForwardDiff())

test_operators(AutoForwardDiff(; chunksize=2));
