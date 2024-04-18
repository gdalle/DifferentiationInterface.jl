dense_second_order_backends = [
    AutoForwardDiff(),  #
    AutoPolyesterForwardDiff(; chunksize=1),  #
    AutoFastDifferentiation(),
    # AutoSymbolics(),
    AutoReverseDiff(),
]

sparse_second_order_backends = [
    AutoSparseForwardDiff(),  #
    AutoSparseFastDifferentiation(),
    # AutoSparseSymbolics(),
]

mixed_second_order_backends = [
    # forward over forward
    SecondOrder(AutoForwardDiff(), AutoEnzyme(Enzyme.Forward)),
    # forward over reverse
    SecondOrder(AutoForwardDiff(), AutoZygote()),
    # reverse over forward
    SecondOrder(AutoEnzyme(Enzyme.Reverse), AutoForwardDiff()),
    # reverse over reverse
    SecondOrder(AutoReverseDiff(), AutoZygote()),
]

##

for backend in vcat(
    dense_second_order_backends, sparse_second_order_backends, mixed_second_order_backends
)
    check_hessian(backend)
end

test_differentiation(
    vcat(dense_second_order_backends, mixed_second_order_backends);
    first_order=false,
    second_order=true,
    logging=get(ENV, "CI", "false") == "false",
);

test_differentiation(
    sparse_second_order_backends;
    first_order=false,
    second_order=true,
    excluded=[HessianScenario],
    logging=get(ENV, "CI", "false") == "false",
);

## only Hessian

only_hessian_backends = [AutoFiniteDiff(), AutoZygote()]

test_differentiation(
    only_hessian_backends;
    first_order=false,
    second_order=true,
    excluded=[SecondDerivativeScenario, HVPScenario],
    logging=get(ENV, "CI", "false") == "false",
);
