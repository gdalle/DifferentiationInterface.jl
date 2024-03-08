using DifferentiationInterface
using Enzyme: Enzyme

test_pullback(EnzymeReverseBackend(), scenarios; type_stability=true);
test_jacobian_and_friends(
    EnzymeReverseBackend(; custom=true), scenarios; type_stability=true
)
test_jacobian_and_friends(
    EnzymeReverseBackend(; custom=false), scenarios; type_stability=true
)
