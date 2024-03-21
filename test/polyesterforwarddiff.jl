using ADTypes: AutoPolyesterForwardDiff
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using PolyesterForwardDiff: PolyesterForwardDiff

using JET: JET
using Test

@test check_available(AutoPolyesterForwardDiff(; chunksize=2))
@test check_mutation(AutoPolyesterForwardDiff(; chunksize=2))
@test check_hessian(AutoPolyesterForwardDiff(; chunksize=2))

test_operators(
    AutoPolyesterForwardDiff(; chunksize=2);
    input_type=Union{Number,AbstractVector},
    output_type=Union{Number,AbstractVector},
    type_stability=false,
);
