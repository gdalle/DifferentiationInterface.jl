using ADTypes: AutoPolyesterForwardDiff
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using PolyesterForwardDiff: PolyesterForwardDiff

using JET: JET
using Test

@test available(AutoPolyesterForwardDiff(; chunksize=2))

test_operators(
    AutoPolyesterForwardDiff(; chunksize=2);
    input_type=Union{Number,AbstractVector},
    output_type=Union{Number,AbstractVector},
    type_stability=false,
);
