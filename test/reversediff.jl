using ADTypes: AutoReverseDiff
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using ReverseDiff: ReverseDiff

using ForwardDiff: ForwardDiff
using JET: JET
using Test

@test available(AutoReverseDiff())
@test supports_mutation(AutoReverseDiff())
@test supports_hessian(AutoReverseDiff())

test_operators(AutoReverseDiff(); type_stability=false);
test_operators(AutoReverseDiff(; compile=true); type_stability=false);
