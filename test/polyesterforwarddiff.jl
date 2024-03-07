using DifferentiationInterface
using PolyesterForwardDiff

# see https://github.com/JuliaDiff/PolyesterForwardDiff.jl/issues/17

test_jacobian_and_friends(
    PolyesterForwardDiffBackend(4; custom=true);
    input_type=Union{Number,AbstractVector},
    output_type=Union{Number,AbstractVector},
    type_stability=true,
);
