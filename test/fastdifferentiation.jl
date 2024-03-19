using DifferentiationInterface: AutoFastDifferentiation
using DifferentiationInterface.DifferentiationTest
using FastDifferentiation: FastDifferentiation
using Test

@test available(AutoFastDifferentiation())

test_operators(
    AutoFastDifferentiation();
    input_type=Union{Number,AbstractVector},
    output_type=Union{Number,AbstractVector},
    second_order=false,
    type_stability=false,
);
