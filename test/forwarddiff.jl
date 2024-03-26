include("test_imports.jl")

using ForwardDiff: ForwardDiff

@test check_available(AutoForwardDiff())
@test check_mutation(AutoForwardDiff())
@test check_hessian(AutoForwardDiff())

test_differentiation(AutoForwardDiff(; chunksize=2); type_stability=false, second_order=true);

test_differentiation(
    AutoForwardDiff(; chunksize=2),
    all_operators(),
    weird_array_scenarios(; static=true, component=true, gpu=false);
    excluded=[jacobian],
);
