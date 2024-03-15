using ADTypes: AutoEnzyme
using Enzyme: Enzyme
using DifferentiationInterface.DifferentiationTest

test_operators_allocating(AutoEnzyme(Enzyme.Reverse));
test_operators_mutating(AutoEnzyme(Enzyme.Reverse));
