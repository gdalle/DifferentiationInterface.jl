include("test_imports.jl")

using ReverseDiff: ReverseDiff

@test check_available(AutoReverseDiff())
@test check_mutation(AutoReverseDiff())
@test check_hessian(AutoReverseDiff())

test_differentiation(AutoReverseDiff());
