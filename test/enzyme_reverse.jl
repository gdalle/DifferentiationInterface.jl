using ADTypes: AutoEnzyme
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using Enzyme: Enzyme
using Test

@test available(AutoEnzyme(Enzyme.Reverse))

test_operators(AutoEnzyme(Enzyme.Reverse); second_order=false);
