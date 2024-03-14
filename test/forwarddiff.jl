using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff
using DifferentiationInterface.DifferentiationTest

test_pushforward(AutoForwardDiff(), default_scenarios(); type_stability=true);
test_derivative(AutoForwardDiff(), default_scenarios(); type_stability=true);
test_multiderivative(AutoForwardDiff(), default_scenarios(); type_stability=true);
test_gradient(AutoForwardDiff(), default_scenarios(); type_stability=false);
test_jacobian(AutoForwardDiff(), default_scenarios(); type_stability=false);
