using DifferentiationInterface
using Enzyme

test_pushforward(EnzymeForwardBackend(), scenarios; type_stability=true);
test_jacobian_and_friends(
    EnzymeForwardBackend(; custom=true), scenarios; type_stability=true
);
test_jacobian_and_friends(
    EnzymeForwardBackend(; custom=false), scenarios; type_stability=true
);
