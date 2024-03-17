using ADTypes: AutoReverseDiff
using ReverseDiff: ReverseDiff
using DifferentiationInterface.DifferentiationTest

test_operators(AutoReverseDiff(); excluded=[:hessian_allocating], type_stability=false);
test_operators(
    AutoReverseDiff(; compile=true); excluded=[:hessian_allocating], type_stability=false
);
