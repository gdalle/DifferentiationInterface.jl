module DifferentiationInterfaceDiffractorExt

import AbstractDifferentiation as AD  # public API for Diffractor
using ADTypes: ADTypes, AutoChainRules, AutoDiffractor
import DifferentiationInterface as DI
using DifferentiationInterface: NoPushforwardExtras
using Diffractor: DiffractorForwardBackend, DiffractorRuleConfig

DI.supports_mutation(::AutoDiffractor) = DI.MutationNotSupported()
DI.mode(::AutoDiffractor) = ADTypes.AbstractForwardMode
DI.mode(::AutoChainRules{<:DiffractorRuleConfig}) = ADTypes.AbstractForwardMode

## Pushforward

DI.prepare_pushforward(f, ::AutoDiffractor, x) = NoPushforwardExtras()

function DI.value_and_pushforward(f, ::AutoDiffractor, x, dx, ::NoPushforwardExtras)
    vpff = AD.value_and_pushforward_function(DiffractorForwardBackend(), f, x)
    y, dy = vpff((dx,))
    return y, dy
end

end
