using DifferentiationInterface
using Enzyme

test_pullback(EnzymeReverseBackend(); output_type=Number, type_stability=true);
test_jacobian_and_friends(
    EnzymeReverseBackend(; custom=true); output_type=Number, type_stability=true
)
test_jacobian_and_friends(
    EnzymeReverseBackend(; custom=false); output_type=Number, type_stability=true
)
