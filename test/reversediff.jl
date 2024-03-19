using ADTypes: AutoReverseDiff
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using ReverseDiff: ReverseDiff

using ForwardDiff: ForwardDiff
using JET: JET
using Test

@test available(AutoReverseDiff())

test_operators(AutoReverseDiff(); second_order=false, type_stability=false);
test_operators(AutoReverseDiff(; compile=true); second_order=false, type_stability=false);
