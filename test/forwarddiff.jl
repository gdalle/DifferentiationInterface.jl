using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff
using DifferentiationInterface.DifferentiationTest

test_operators(AutoForwardDiff(; chunksize=2));
