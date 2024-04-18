test_differentiation(AutoForwardDiff(); logging=get(ENV, "CI", "false") == "false")

test_differentiation(
    AutoSparseForwardDiff(),
    sparse_scenarios();
    sparsity=true,
    logging=get(ENV, "CI", "false") == "false",
)
