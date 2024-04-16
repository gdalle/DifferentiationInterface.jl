module DifferentiationInterfaceDiffractorExt

using ADTypes: ADTypes, AutoChainRules, AutoDiffractor
import DifferentiationInterface as DI
using DifferentiationInterface: NoPushforwardExtras
using Diffractor: DiffractorRuleConfig, TaylorTangentIndex, ZeroBundle, bundle, ∂☆

DI.check_available(::AutoDiffractor) = true
DI.supports_mutation(::AutoDiffractor) = DI.MutationNotSupported()
DI.mode(::AutoDiffractor) = ADTypes.AbstractForwardMode
DI.mode(::AutoChainRules{<:DiffractorRuleConfig}) = ADTypes.AbstractForwardMode

## Pushforward

DI.prepare_pushforward(f, ::AutoDiffractor, x) = NoPushforwardExtras()

function DI.pushforward(f, ::AutoDiffractor, x, dx, ::NoPushforwardExtras)
    # code copied from Diffractor.jl
    z = ∂☆{1}()(ZeroBundle{1}(f), bundle(x, dx))
    dy = z[TaylorTangentIndex(1)]
    return dy
end

function DI.value_and_pushforward(
    f, backend::AutoDiffractor, x, dx, extras::NoPushforwardExtras
)
    return f(x), DI.pushforward(f, backend, x, dx, extras)
end

end
