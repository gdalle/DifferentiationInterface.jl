using ADTypes: AutoReverseDiff
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using ReverseDiff: ReverseDiff

using ForwardDiff: ForwardDiff
using JET: JET
using Test

@test_broken check_available(AutoReverseDiff())
@test check_mutation(AutoReverseDiff())

test_operators(AutoReverseDiff(); input_type=AbstractArray, type_stability=false);
