module DifferentiationInterfaceTrackerExt

using ADTypes: AutoTracker
import DifferentiationInterface as DI
using DifferentiationInterface: myupdate!!
using Tracker: Tracker, back, data, forward, gradient, jacobian, param, withgradient

DI.supports_mutation(::AutoTracker) = DI.MutationNotSupported()

## Pullback

function DI.value_and_pullback(f::F, ::AutoTracker, x, dy, extras::Nothing) where {F}
    y, back = forward(f, x)
    return y, data(only(back(dy)))
end

function DI.value_and_pullback!!(f::F, dx, backend::AutoTracker, x, dy, extras) where {F}
    y, new_dx = DI.value_and_pullback(f, backend, x, dy, extras)
    return y, myupdate!!(dx, new_dx)
end

end
