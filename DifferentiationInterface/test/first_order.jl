dense_backends = [
    AutoChainRules(Zygote.ZygoteRuleConfig()),
    AutoDiffractor(),
    AutoEnzyme(Enzyme.Forward),
    AutoEnzyme(Enzyme.Reverse),
    AutoFastDifferentiation(),
    AutoFiniteDiff(),
    AutoFiniteDifferences(FiniteDifferences.central_fdm(3, 1)),
    AutoForwardDiff(),
    AutoPolyesterForwardDiff(; chunksize=1),
    AutoReverseDiff(; compile=true),
    AutoSymbolics(),
    AutoTapir(),
    AutoTracker(),
    AutoZygote(),
]

sparse_backends = [
    AutoSparse(AutoFastDifferentiation()),
    AutoSparse(AutoForwardDiff()),
    AutoSparse(AutoSymbolics()),
]

##

for backend in vcat(dense_backends, sparse_backends)
    @test check_available(backend)
end

test_differentiation(
    dense_backends; second_order=false, logging=get(ENV, "CI", "false") == "false"
);

test_differentiation(
    sparse_backends;
    second_order=false,
    excluded=[JacobianScenario],
    logging=get(ENV, "CI", "false") == "false",
);
