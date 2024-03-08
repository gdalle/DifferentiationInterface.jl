using DifferentiationInterface
using ForwardDiff: ForwardDiff

test_pushforward(ForwardDiffBackend(), scenarios; type_stability=true);
test_jacobian_and_friends(
    ForwardDiffBackend(; custom=true), scenarios; type_stability=false
);
test_jacobian_and_friends(
    ForwardDiffBackend(; custom=false), scenarios; type_stability=true
);
