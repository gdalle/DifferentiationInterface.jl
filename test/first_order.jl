##

using Diffractor: Diffractor
using Enzyme: Enzyme
using FastDifferentiation: FastDifferentiation
using FiniteDiff: FiniteDiff
using FiniteDifferences: FiniteDifferences
using ForwardDiff: ForwardDiff
using PolyesterForwardDiff: PolyesterForwardDiff
using ReverseDiff: ReverseDiff
using SparseDiffTools: SparseDiffTools
using Symbolics: Symbolics
using Tapir: Tapir
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
    AutoSparseFiniteDiff(),
    AutoFiniteDifferences(FiniteDifferences.central_fdm(3, 1)),
    AutoForwardDiff(),
    AutoSparseForwardDiff(),
    AutoPolyesterForwardDiff(; chunksize=2),
    # AutoSparsePolyesterForwardDiff(; chunksize=2),
    AutoReverseDiff(),
    # AutoSparseReverseDiff(),
    # AutoTapir(),
    AutoTracker(),
    AutoZygote(),
    # AutoSparseZygote(),
]

##

for backend in all_backends
    @test check_available(backend)
end

test_differentiation(all_backends; second_order=false, logging=true);
