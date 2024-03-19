using ADTypes: AutoFiniteDiff
using FiniteDiff: FiniteDiff
using DifferentiationInterface.DifferentiationTest
using Test

@test available(AutoFiniteDiff())

test_operators(AutoFiniteDiff(); second_order=false, excluded=[:jacobian_allocating]);
test_operators(AutoFiniteDiff(), [:jacobian_allocating]; type_stability=false);
