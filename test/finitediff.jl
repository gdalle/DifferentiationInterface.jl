using ADTypes: AutoFiniteDiff
using FiniteDiff: FiniteDiff
using DifferentiationInterface.DifferentiationTest

test_pushforward(AutoFiniteDiff(), default_scenarios(); type_stability=true);
test_derivative(AutoFiniteDiff(), default_scenarios(); type_stability=true);
test_multiderivative(AutoFiniteDiff(), default_scenarios(); type_stability=true);
test_gradient(AutoFiniteDiff(), default_scenarios(); type_stability=true);
test_jacobian(AutoFiniteDiff(), default_scenarios(); type_stability=false);
