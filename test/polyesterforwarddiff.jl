using ADTypes: AutoPolyesterForwardDiff
using PolyesterForwardDiff: PolyesterForwardDiff

polyesterforwarddiff_backend = AutoPolyesterForwardDiff(; chunksize=2)

test_pushforward(polyesterforwarddiff_backend, scenarios; type_stability=false);

test_jacobian_and_friends(
    polyesterforwarddiff_backend,
    scenarios;
    input_type=Union{Number,AbstractVector},
    output_type=Union{Number,AbstractVector},
    type_stability=false,
);
