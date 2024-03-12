module DifferentiationInterfaceDiffractorExt

import AbstractDifferentiation as AD  # public API for Diffractor
using ADTypes: AutoChainRules, AutoDiffractor
using DifferentiationInterface: update!
import DifferentiationInterface as DI
using Diffractor: DiffractorForwardBackend, DiffractorRuleConfig
using DocStringExtensions

DI.autodiff_mode(::AutoDiffractor) = DI.ForwardMode()
DI.autodiff_mode(::AutoChainRules{<:DiffractorRuleConfig}) = DI.ForwardMode()

function DI.value_and_pushforward(::AutoDiffractor, f, x, dx, extras::Nothing=nothing)
    vpff = AD.value_and_pushforward_function(DiffractorForwardBackend(), f, x)
    y, dy = vpff((dx,))
    return y, dy
end

function DI.value_and_pushforward!(dy, ::AutoDiffractor, f, x, dx, extras::Nothing=nothing)
    vpff = AD.value_and_pushforward_function(DiffractorForwardBackend(), f, x)
    y, new_dy = vpff((dx,))
    return y, update!(dy, new_dy)
end

end
