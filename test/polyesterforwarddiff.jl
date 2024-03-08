using ADTypes: AutoPolyesterForwardDiff
using PolyesterForwardDiff: PolyesterForwardDiff

# see https://github.com/JuliaDiff/PolyesterForwardDiff.jl/issues/17

test_pushforward(AutoPolyesterForwardDiff(; chunksize=4), scenarios; type_stability=false);

test_jacobian_and_friends(
    AutoPolyesterForwardDiff(; chunksize=4),
    scenarios;
    input_type=Union{Number,AbstractVector},
    output_type=Union{Number,AbstractVector},
    type_stability=false,
);
