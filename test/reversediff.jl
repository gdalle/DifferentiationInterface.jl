using DifferentiationInterface
using ReverseDiff

test_pullback(ReverseDiffBackend(); input_type=AbstractArray, type_stability=false)
