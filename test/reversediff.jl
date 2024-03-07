using DifferentiationInterface
using ReverseDiff

test_pullback(ReverseDiffBackend(); type_stability=false);
test_jacobian_and_friends(ReverseDiffBackend(; custom=true); type_stability=false);
test_jacobian_and_friends(ReverseDiffBackend(; custom=false); type_stability=false);
