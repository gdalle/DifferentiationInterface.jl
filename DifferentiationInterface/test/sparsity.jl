sparse_backends = [
    AutoSparseFastDifferentiation(),
    AutoSparseForwardDiff(),
    AutoSparseFiniteDiff(),
    AutoSparseSymbolics(),
    AutoSparseZygote(),
]

sparse_second_order_backends = [
    AutoSparseFastDifferentiation(),
    SecondOrder(AutoSparseForwardDiff(), AutoZygote()),
    SecondOrder(AutoSparseFiniteDiff(), AutoZygote()),
]

for backend in vcat(sparse_backends, sparse_second_order_backends)
    @test check_available(backend)
end

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
