using DifferentiationInterface
using ReverseDiff

test_pullback(ReverseDiffBackend(); input_type=AbstractArray, type_stability=false);
test_jacobian_and_friends(
    ReverseDiffBackend(); input_type=AbstractArray, type_stability=false
);
