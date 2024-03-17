using DifferentiationInterface: AutoFastDifferentiation
using DifferentiationInterface.DifferentiationTest
using FastDifferentiation: FastDifferentiation

test_operators_allocating(
    AutoFastDifferentiation();
    excluded=[:pushforward, :pullback],
    input_type=Union{Number,AbstractVector},
    output_type=Union{Number,AbstractVector},
    type_stability=false,
);
