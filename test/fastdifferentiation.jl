include("test_imports.jl")

using DifferentiationInterface: AutoFastDifferentiation
using FastDifferentiation: FastDifferentiation

@test check_available(AutoFastDifferentiation())
@test !check_mutation(AutoFastDifferentiation())
@test_broken !check_hessian(AutoFastDifferentiation())

test_differentiation(
    AutoFastDifferentiation();
    input_type=Union{Number,AbstractVector},
    output_type=Union{Number,AbstractVector},
    second_order=false,
);
