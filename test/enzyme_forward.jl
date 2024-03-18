using ADTypes: AutoEnzyme
using Enzyme: Enzyme
using DifferentiationInterface.DifferentiationTest

test_operators(
    AutoEnzyme(Enzyme.Forward); second_order=false, excluded=[:jacobian_allocating]
);
test_operators(AutoEnzyme(Enzyme.Forward), [:jacobian_allocating]; type_stability=false);
