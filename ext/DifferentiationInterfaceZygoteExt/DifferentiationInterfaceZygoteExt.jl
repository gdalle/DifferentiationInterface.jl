module DifferentiationInterfaceZygoteExt

using ADTypes: AutoZygote
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

## Gradient

function DI.value_and_gradient(f::F, ::AutoZygote, x, extras::Nothing) where {F}
    (; val, grad) = withgradient(f, x)
    return val, only(grad)
end

function DI.gradient(f::F, ::AutoZygote, x, extras::Nothing) where {F}
    return only(gradient(f, x))
end

end
