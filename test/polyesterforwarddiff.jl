using DifferentiationInterface
using PolyesterForwardDiff: PolyesterForwardDiff

# see https://github.com/JuliaDiff/PolyesterForwardDiff.jl/issues/17

test_pushforward(
    PolyesterForwardDiffBackend(4; custom=true), scenarios; type_stability=false
);

test_jacobian_and_friends(
    PolyesterForwardDiffBackend(4; custom=true),
    scenarios;
    input_type=Union{Number,AbstractVector},
    output_type=Union{Number,AbstractVector},
    type_stability=false,
);
