module DifferentiationInterfaceDiffractorExt

using ADTypes: ADTypes, AutoDiffractor
import DifferentiationInterface as DI
using DifferentiationInterface: NoPushforwardExtras, Tangents
using Diffractor: DiffractorRuleConfig, TaylorTangentIndex, ZeroBundle, bundle, ∂☆

DI.check_available(::AutoDiffractor) = true
DI.twoarg_support(::AutoDiffractor) = DI.TwoArgNotSupported()
DI.pullback_performance(::AutoDiffractor) = DI.PullbackSlow()

## Pushforward

DI.prepare_pushforward(f, ::AutoDiffractor, x, tx::Tangents{1}) = NoPushforwardExtras()

function DI.pushforward(f, ::AutoDiffractor, x, tx::Tangents{1}, ::NoPushforwardExtras)
    dx = only(tx)
    # code copied from Diffractor.jl
    z = ∂☆{1}()(ZeroBundle{1}(f), bundle(x, dx))
    dy = z[TaylorTangentIndex(1)]
    return Tangents(dy)
end

function DI.value_and_pushforward(
    f, backend::AutoDiffractor, x, tx::Tangents{1}, extras::NoPushforwardExtras
)
    return f(x), DI.pushforward(f, backend, x, tx, extras)
end

end
