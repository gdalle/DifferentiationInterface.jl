using ADTypes: AutoPolyesterForwardDiff
using PolyesterForwardDiff: PolyesterForwardDiff
using DifferentiationInterface.DifferentiationTest

test_operators_allocating(
    AutoPolyesterForwardDiff(; chunksize=2);
    input_type=Union{Number,AbstractVector},
    output_type=Union{Number,AbstractVector},
    type_stability=false,
);

test_operators_mutating(
    AutoPolyesterForwardDiff(; chunksize=2);
    input_type=Union{Number,AbstractVector},
    output_type=Union{Number,AbstractVector},
    type_stability=false,
);

test_second_order_operators_allocating(
    AutoPolyesterForwardDiff(; chunksize=2);
    input_type=Union{Number,AbstractVector},
    type_stability=false,
)
