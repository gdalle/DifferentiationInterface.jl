using DifferentiationInterface
using FiniteDiff

test_pushforward(FiniteDiffBackend(); type_stability=true);
test_jacobian_and_friends(FiniteDiffBackend(; custom=true); type_stability=false);
test_jacobian_and_friends(FiniteDiffBackend(; custom=false); type_stability=true);
