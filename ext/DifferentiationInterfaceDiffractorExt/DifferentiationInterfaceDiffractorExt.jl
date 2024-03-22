module DifferentiationInterfaceDiffractorExt

import AbstractDifferentiation as AD  # public API for Diffractor
using ADTypes: ADTypes, AutoChainRules, AutoDiffractor
using DifferentiationInterface: myupdate!
import DifferentiationInterface as DI
using Diffractor: DiffractorForwardBackend, DiffractorRuleConfig

DI.supports_mutation(::AutoDiffractor) = DI.MutationNotSupported()
DI.mode(::AutoDiffractor) = ADTypes.AbstractForwardMode
DI.mode(::AutoChainRules{<:DiffractorRuleConfig}) = ADTypes.AbstractForwardMode

function DI.value_and_pushforward(::AutoDiffractor, f, x, dx)
    vpff = AD.value_and_pushforward_function(DiffractorForwardBackend(), f, x)
    y, dy = vpff((dx,))
    return y, dy
end

function DI.value_and_pushforward!(
    dy::Union{Number,AbstractArray}, ::AutoDiffractor, f, x, dx
)
    vpff = AD.value_and_pushforward_function(DiffractorForwardBackend(), f, x)
    y, new_dy = vpff((dx,))
    return y, myupdate!(dy, new_dy)
end

end
