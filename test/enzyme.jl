using DifferentiationInterface
using Enzyme

test_pushforward(EnzymeForwardBackend(); type_stability=true)
test_pullback(EnzymeReverseBackend(); output_type=Number, type_stability=true)
