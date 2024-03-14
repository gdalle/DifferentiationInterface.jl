using ADTypes: AutoPolyesterForwardDiff
using PolyesterForwardDiff: PolyesterForwardDiff
using DifferentiationInterface.DifferentiationTest

test_all_operators(
    AutoPolyesterForwardDiff(; chunksize=2),
    default_scenarios();
    input_type=Union{Number,AbstractVector},
    output_type=Union{Number,AbstractVector},
    type_stability=false,
);
