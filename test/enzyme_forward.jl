using ADTypes: AutoEnzyme
using Enzyme: Enzyme
using DifferentiationInterface.DifferentiationTest

test_pushforward(AutoEnzyme(Val(:forward)), default_scenarios());
test_derivative(AutoEnzyme(Val(:forward)), default_scenarios());
test_multiderivative(AutoEnzyme(Val(:forward)), default_scenarios());
test_gradient(AutoEnzyme(Val(:forward)), default_scenarios());
test_jacobian(AutoEnzyme(Val(:forward)), default_scenarios(); type_stability=false);
