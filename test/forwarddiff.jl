using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff
using DifferentiationInterface.DifferentiationTest

test_operators_allocating(AutoForwardDiff(; chunksize=2));
test_operators_mutating(AutoForwardDiff(; chunksize=2));

test_second_order_operators_allocating(AutoForwardDiff(; chunksize=2))
