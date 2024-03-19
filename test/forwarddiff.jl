using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff
using DifferentiationInterface.DifferentiationTest
using Test

@test available(AutoForwardDiff())

test_operators(AutoForwardDiff(; chunksize=2));
