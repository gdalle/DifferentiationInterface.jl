all_backends = [
    AutoChainRules(Zygote.ZygoteRuleConfig()),
    # AutoDiffractor(),
    # AutoEnzyme(Enzyme.Forward),
    # AutoEnzyme(Enzyme.Reverse),
    # AutoFastDifferentiation(),
    # AutoFiniteDiff(),
    # AutoFiniteDifferences(FiniteDifferences.central_fdm(3, 1)),
    # AutoForwardDiff(),
    # AutoPolyesterForwardDiff(; chunksize=1),
    # AutoReverseDiff(; compile=true),
    # AutoTapir(),
    # AutoTracker(),
    # AutoZygote(),
]

##

for backend in all_backends
    @test check_available(backend)
end

test_differentiation(
    all_backends; second_order=false, logging=get(ENV, "CI", "false") == "false"
);
