using ADTypes: AutoFiniteDiff
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using FiniteDiff: FiniteDiff

using ForwardDiff: ForwardDiff
using JET: JET
using Test

@test check_available(AutoFiniteDiff())
@test check_mutation(AutoFiniteDiff())
@test !check_hessian(AutoFiniteDiff())

test_operators(AutoFiniteDiff(); second_order=false, excluded=[:jacobian_allocating]);
test_operators(AutoFiniteDiff(), [:jacobian_allocating]; type_stability=false);
