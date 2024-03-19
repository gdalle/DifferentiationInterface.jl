using ADTypes: AutoEnzyme
using Enzyme: Enzyme
using DifferentiationInterface.DifferentiationTest
using Test

@test available(AutoEnzyme(Enzyme.Reverse))

test_operators(AutoEnzyme(Enzyme.Reverse); second_order=false);
