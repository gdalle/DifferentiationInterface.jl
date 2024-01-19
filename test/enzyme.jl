using DifferentiationInterface
using Enzyme

test_pushforward(EnzymeBackend(); type_stability=true)
test_pullback(EnzymeBackend(); output_type=Number, type_stability=true)
