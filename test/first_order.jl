##

using Diffractor: Diffractor
using Enzyme: Enzyme
using FastDifferentiation: FastDifferentiation
using FiniteDiff: FiniteDiff
using FiniteDifferences: FiniteDifferences
using ForwardDiff: ForwardDiff
using PolyesterForwardDiff: PolyesterForwardDiff
using ReverseDiff: ReverseDiff
using Tracker: Tracker
using Zygote: Zygote

##

all_backends = [
    AutoChainRules(Zygote.ZygoteRuleConfig()),
    AutoDiffractor(),
    AutoEnzyme(Enzyme.Forward),
    AutoEnzyme(Enzyme.Reverse),
    AutoFastDifferentiation(),
    AutoFiniteDiff(),
    AutoFiniteDifferences(FiniteDifferences.central_fdm(3, 1)),
    AutoForwardDiff(),
    AutoPolyesterForwardDiff(; chunksize=2),
    AutoReverseDiff(),
    AutoTracker(),
    AutoZygote(),
]

for backend in all_backends
    @test check_available(backend)
end

test_differentiation(all_backends; second_order=false);
