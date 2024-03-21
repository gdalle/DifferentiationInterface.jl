using ADTypes: AutoEnzyme
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using Enzyme: Enzyme

using ForwardDiff: ForwardDiff
using JET: JET
using Test

@test check_available(AutoEnzyme(Enzyme.Forward))
@test check_mutation(AutoEnzyme(Enzyme.Forward))
@test !check_hessian(AutoEnzyme(Enzyme.Forward))

test_operators(
    AutoEnzyme(Enzyme.Forward); second_order=false, excluded=[:jacobian_allocating]
);
test_operators(AutoEnzyme(Enzyme.Forward), [:jacobian_allocating]; type_stability=false);
