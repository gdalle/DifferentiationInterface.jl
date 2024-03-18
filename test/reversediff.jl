using ADTypes: AutoReverseDiff
using ReverseDiff: ReverseDiff
using DifferentiationInterface.DifferentiationTest

test_operators(AutoReverseDiff(); second_order=false, type_stability=false);
test_operators(AutoReverseDiff(; compile=true); second_order=false, type_stability=false);
