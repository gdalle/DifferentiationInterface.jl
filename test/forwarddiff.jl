using ADTypes: AutoForwardDiff
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using ForwardDiff: ForwardDiff
using Test

@test available(AutoForwardDiff())

test_operators(AutoForwardDiff(; chunksize=2));
