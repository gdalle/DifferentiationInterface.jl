test_differentiation(
    AutoSparse(AutoSymbolics()),
    sparse_scenarios();
    sparsity=true,
    logging=get(ENV, "CI", "false") == "false",
)
