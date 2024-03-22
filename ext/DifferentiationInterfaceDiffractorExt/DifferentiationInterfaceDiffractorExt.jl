module DifferentiationInterfaceDiffractorExt

import AbstractDifferentiation as AD  # public API for Diffractor
using ADTypes: ADTypes, AutoChainRules, AutoDiffractor
using DifferentiationInterface: myupdate!
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

function DI.value_and_pushforward!(
    f::F, dy, backend::AutoDiffractor, x, dx, extras
) where {F}
    y, new_dy = DI.value_and_pushforward(f, backend, x, dx, extras)
    return y, myupdate!(dy, new_dy)
end

end
