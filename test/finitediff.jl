using DifferentiationInterface
using FiniteDiff

test_pushforward(FiniteDiffBackend(), scenarios; type_stability=true);
test_jacobian_and_friends(FiniteDiffBackend(; custom=true), scenarios; type_stability=false);
test_jacobian_and_friends(FiniteDiffBackend(; custom=false), scenarios; type_stability=true);
