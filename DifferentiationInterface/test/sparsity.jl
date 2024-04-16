sparse_backends = [
    AutoSparseFastDifferentiation(),
    AutoSparseForwardDiff(),
    AutoSparseFiniteDiff(),
    AutoSparseZygote(),
]

sparse_second_order_backends = [
    AutoSparseFastDifferentiation(),
    SecondOrder(AutoSparseForwardDiff(), AutoZygote()),
    SecondOrder(AutoSparseFiniteDiff(), AutoZygote()),
]

test_differentiation(
    sparse_backends,
    sparse_scenarios(rand(5));
    sparsity=true,
    second_order=false,
    logging=get(ENV, "CI", "false") == "false",
)

test_differentiation(
    sparse_second_order_backends,
    sparse_scenarios(rand(5));
    sparsity=true,
    first_order=false,
    logging=get(ENV, "CI", "false") == "false",
)
