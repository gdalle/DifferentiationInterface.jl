using DifferentiationInterface
using ForwardDiff

test_pushforward(ForwardDiffBackend(); type_stability=true);
test_jacobian_and_friends(ForwardDiffBackend(; custom=true); type_stability=false);
test_jacobian_and_friends(ForwardDiffBackend(; custom=false); type_stability=true);
