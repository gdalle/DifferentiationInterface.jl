include("test_imports.jl")

using PolyesterForwardDiff: PolyesterForwardDiff

@test check_available(AutoPolyesterForwardDiff(; chunksize=2))
@test check_mutation(AutoPolyesterForwardDiff(; chunksize=2))
@test check_hessian(AutoPolyesterForwardDiff(; chunksize=2))

test_differentiation(
    AutoPolyesterForwardDiff(; chunksize=2);
    input_type=Union{Number,AbstractVector},
    output_type=Union{Number,AbstractVector},
    type_stability=false,
);
