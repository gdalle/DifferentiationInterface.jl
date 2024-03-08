using DifferentiationInterface
using ReverseDiff

test_pullback(ReverseDiffBackend(), scenarios; type_stability=false);
test_jacobian_and_friends(
    ReverseDiffBackend(; custom=true), scenarios; type_stability=false
);
test_jacobian_and_friends(
    ReverseDiffBackend(; custom=false), scenarios; type_stability=false
);
