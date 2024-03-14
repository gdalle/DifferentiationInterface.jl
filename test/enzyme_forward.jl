using ADTypes: AutoEnzyme
using Enzyme: Enzyme
using DifferentiationInterface.DifferentiationTest

test_all_operators(AutoEnzyme(Val(:forward)), default_scenarios(); type_stability=false);
