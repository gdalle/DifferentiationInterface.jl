module DifferentiationInterfaceDiffractorExt

using ADTypes: ADTypes, AutoDiffractor
import DifferentiationInterface as DI
using DifferentiationInterface: NoPushforwardPrep, Tangents
using Diffractor: DiffractorRuleConfig, TaylorTangentIndex, ZeroBundle, bundle, ∂☆

DI.check_available(::AutoDiffractor) = true
DI.inplace_support(::AutoDiffractor) = DI.InPlaceNotSupported()
DI.pullback_performance(::AutoDiffractor) = DI.PullbackSlow()

## Pushforward

DI.prepare_pushforward(f, ::AutoDiffractor, x, tx::Tangents) = NoPushforwardPrep()

function DI.pushforward(f, ::NoPushforwardPrep, ::AutoDiffractor, x, tx::Tangents)
    ty = map(tx) do dx
        # code copied from Diffractor.jl
        z = ∂☆{1}()(ZeroBundle{1}(f), bundle(x, dx))
        dy = z[TaylorTangentIndex(1)]
    end
    return ty
end

function DI.value_and_pushforward(
    f, prep::NoPushforwardPrep, backend::AutoDiffractor, x, tx::Tangents
)
    return f(x), DI.pushforward(f, prep, backend, x, tx)
end

end
