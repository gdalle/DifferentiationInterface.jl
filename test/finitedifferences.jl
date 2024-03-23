using ADTypes: AutoFiniteDifferences
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using FiniteDifferences: FiniteDifferences, central_fdm

using ForwardDiff: ForwardDiff
using JET: JET
using Test

@test check_available(AutoFiniteDifferences(central_fdm(5, 1)))
@test !check_mutation(AutoFiniteDifferences(central_fdm(5, 1)))
@test_broken !check_hessian(AutoFiniteDifferences(central_fdm(5, 1)))

test_operators(
    AutoFiniteDifferences(central_fdm(5, 1)); second_order=false, type_stability=false
);
