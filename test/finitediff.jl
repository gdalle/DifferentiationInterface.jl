include("test_imports.jl")

using FiniteDiff: FiniteDiff

@test check_available(AutoFiniteDiff())
@test check_mutation(AutoFiniteDiff())
@test_broken check_hessian(AutoFiniteDiff())

test_differentiation(AutoFiniteDiff(); second_order=false);
