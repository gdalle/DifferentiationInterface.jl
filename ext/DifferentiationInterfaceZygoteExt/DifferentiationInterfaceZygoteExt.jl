module DifferentiationInterfaceZygoteExt

using ADTypes: AutoZygote
using DifferentiationInterface: myupdate!!
import DifferentiationInterface as DI
using DocStringExtensions
using Zygote: ZygoteRuleConfig, gradient, jacobian, pullback, withgradient, withjacobian

DI.supports_mutation(::AutoZygote) = DI.MutationNotSupported()

## Pullback

function DI.value_and_pullback(f::F, ::AutoZygote, x, dy, extras::Nothing) where {F}
    y, back = pullback(f, x)
    dx = only(back(dy))
    return y, dx
end

function DI.value_and_pullback!!(f::F, dx, backend::AutoZygote, x, dy, extras) where {F}
    y, new_dx = DI.value_and_pullback(f, backend, x, dy, extras)
    return y, myupdate!!(dx, new_dx)
end

end
