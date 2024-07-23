module DifferentiationInterfaceStochasticADExt

import DifferentiationInterface as DI
import ADTypes
using DifferentiationInterface:
    DerivativeExtras, GradientExtras, HessianExtras, JacobianExtras, NoPullbackExtras
using DocStringExtensions
using StochasticAD: derivative_estimate

DI.check_available(::AutoStochasticAD) = true
DI.twoarg_support(::AutoStochasticAD) = DI.TwoArgNotSupported()
DI.pullback_performance(::AutoStochasticAD) = DI.PushforwardSlow()

DI.prepare_pushforward(f, ::AutoStochasticAD, x, dx) = NoPushforwardExtras()

function DI.pushforward(
    f, ad::AutoStochasticAD, x, dx, ::NoPushforwardExtras)
    return mean(derivative_estimate(f, x, direction=dx) for _ in 1:ad.n_samples)
end

function DI.value_and_pushforward(
    f, backend::AutoStochasticAD, x, dx, extras::NoPushforwardExtras
)
    return f(x), DI.pushforward(f, backend, x, dx, extras)
end

end # module
