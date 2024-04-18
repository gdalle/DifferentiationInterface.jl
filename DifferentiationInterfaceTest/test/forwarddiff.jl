test_differentiation(AutoForwardDiff(); logging=get(ENV, "CI", "false") == "false")

#=
test_differentiation(
    AutoSparse(AutoForwardDiff()),
    sparse_scenarios();
    sparsity=true,
    logging=get(ENV, "CI", "false") == "false",
)
=#

test_differentiation(
    AutoForwardDiff(),
    component_scenarios();
    excluded=[HessianScenario],
    logging=get(ENV, "CI", "false") == "false",
)
