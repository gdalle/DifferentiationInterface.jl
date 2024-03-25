include("test_imports.jl")

using Diffractor: Diffractor

@test check_available(AutoDiffractor())
@test !check_mutation(AutoDiffractor())
@test_broken check_hessian(AutoDiffractor())

test_differentiation(AutoDiffractor(); second_order=false);
