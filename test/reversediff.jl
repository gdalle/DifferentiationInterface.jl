using ADTypes: AutoReverseDiff
using ReverseDiff: ReverseDiff
using DifferentiationInterface.DifferentiationTest

test_operators_allocating(AutoReverseDiff(); type_stability=false);
test_operators_mutating(AutoReverseDiff(); type_stability=false);

test_operators_allocating(AutoReverseDiff(; compile=true); type_stability=false);
test_operators_mutating(AutoReverseDiff(; compile=true); type_stability=false);

test_second_order_operators_allocating(
    AutoReverseDiff(); excluded=[:hessian], type_stability=false
)
