module DifferentiationInterfaceDiffractorExt

import AbstractDifferentiation as AD  # public API for Diffractor
using ADTypes: ADTypes, AutoChainRules, AutoDiffractor
import DifferentiationInterface as DI
using Diffractor: DiffractorForwardBackend, DiffractorRuleConfig

DI.supports_mutation(::AutoDiffractor) = DI.MutationNotSupported()
DI.mode(::AutoDiffractor) = ADTypes.AbstractForwardMode
DI.mode(::AutoChainRules{<:DiffractorRuleConfig}) = ADTypes.AbstractForwardMode

function DI.value_and_pushforward(f::F, ::AutoDiffractor, x, dx, extras::Nothing) where {F}
    vpff = AD.value_and_pushforward_function(DiffractorForwardBackend(), f, x)
    y, dy = vpff((dx,))
    return y, dy
end

end
