using ADTypes: AutoEnzyme
using Enzyme: Enzyme
using DifferentiationInterface.DifferentiationTest

test_operators_allocating(AutoEnzyme(Enzyme.Forward); excluded=[:jacobian]);
test_operators_allocating(
    AutoEnzyme(Enzyme.Forward); included=[:jacobian], type_stability=false
);

test_operators_mutating(AutoEnzyme(Enzyme.Forward));
