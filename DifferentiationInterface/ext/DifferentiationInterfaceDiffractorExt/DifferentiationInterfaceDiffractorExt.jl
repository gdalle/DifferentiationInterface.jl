module DifferentiationInterfaceDiffractorExt

using ADTypes: ADTypes, AutoDiffractor
import DifferentiationInterface as DI
using DifferentiationInterface: NoPushforwardExtras
using Diffractor: DiffractorRuleConfig, TaylorTangentIndex, ZeroBundle, bundle, ∂☆

DI.check_available(::AutoDiffractor) = true
DI.mutation_support(::AutoDiffractor) = DI.MutationNotSupported()
DI.pullback_performance(::AutoDiffractor) = DI.PullbackSlow()

## Pushforward

DI.prepare_pushforward(f, ::AutoDiffractor, x, dx) = NoPushforwardExtras()

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
