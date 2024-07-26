module DifferentiationInterfaceForwardDiffReverseDiffExt

using ADTypes: AutoForwardDiff, AutoReverseDiff
import DifferentiationInterface as DI
using DifferentiationInterface:
    Batch,
    SecondOrder,
    gradient,
    hvp,
    inner,
    outer,
    prepare_gradient,
    prepare_hvp,
    prepare_hvp_batched,
    prepare_pushforward,
    prepare_pushforward_batched,
    PreparedInnerGradient
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff

end
