using ADTypes: AutoReverseDiff
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using ReverseDiff: ReverseDiff

using ForwardDiff: ForwardDiff
using JET: JET
using Test

@test check_available(AutoReverseDiff())
@test check_mutation(AutoReverseDiff())

test_operators(AutoReverseDiff(); type_stability=false);
test_operators(AutoReverseDiff(; compile=true); type_stability=false);
