module DifferentiationInterfaceDiffractorExt

using ADTypes: ADTypes, AutoDiffractor
import DifferentiationInterface as DI
using DifferentiationInterface: NoPushforwardExtras, Tangents
using Diffractor: DiffractorRuleConfig, TaylorTangentIndex, ZeroBundle, bundle, ∂☆

DI.check_available(::AutoDiffractor) = true
DI.inplace_support(::AutoDiffractor) = DI.InPlaceNotSupported()
DI.pullback_performance(::AutoDiffractor) = DI.PullbackSlow()

## Pushforward

DI.prepare_pushforward(f, ::AutoDiffractor, x, tx::Tangents) = NoPushforwardExtras()

function DI.pushforward(f, ::NoPushforwardExtras, ::AutoDiffractor, x, tx::Tangents)
    dys = map(tx.d) do dx
        # code copied from Diffractor.jl
        z = ∂☆{1}()(ZeroBundle{1}(f), bundle(x, dx))
        dy = z[TaylorTangentIndex(1)]
    end
    return Tangents(dys)
end

function DI.value_and_pushforward(
    f, extras::NoPushforwardExtras, backend::AutoDiffractor, x, tx::Tangents
)
    return f(x), DI.pushforward(f, extras, backend, x, tx)
end

end
