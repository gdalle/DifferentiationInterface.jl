using ADTypes: AutoFiniteDiff
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using FiniteDiff: FiniteDiff
using Test

@test available(AutoFiniteDiff())

test_operators(AutoFiniteDiff(); second_order=false, excluded=[:jacobian_allocating]);
test_operators(AutoFiniteDiff(), [:jacobian_allocating]; type_stability=false);
