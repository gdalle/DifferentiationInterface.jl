module DifferentiationInterfaceStochasticADExt

import DifferentiationInterface as DI
using DifferentiationInterface: AutoStochasticAD
using DifferentiationInterface: NoPushforwardExtras, NoPullbackExtras
using StochasticAD: derivative_estimate

DI.check_available(::AutoStochasticAD) = true
DI.twoarg_support(::AutoStochasticAD) = DI.TwoArgNotSupported()
DI.pullback_performance(::AutoStochasticAD) = DI.PushforwardSlow()

DI.prepare_pushforward(f, ::AutoStochasticAD, x, dx) = NoPushforwardExtras()
DI.prepare_pullback(f, ::AutoStochasticAD, x, dx) = NoPullbackExtras()

function DI.pushforward(
    f, ad::AutoStochasticAD, x, dx, ::NoPushforwardExtras)
    return sum(derivative_estimate(f, x, direction=dx) for _ in 1:ad.n_samples) / ad.n_samples
end

function DI.value_and_pushforward(
    f, backend::AutoStochasticAD, x, dx, extras::NoPushforwardExtras
)
    return f(x), DI.pushforward(f, backend, x, dx, extras)
end

function DI.pullback(
    f, backend::AutoStochasticAD, x, dy, extras::NoPullbackExtras
)
    jacobian = reduce(hcat, sum(derivative_estimate(f, x) for _ in 1:backend.n_samples) / backend.n_samples)
    return jacobian' * dy
end

function DI.value_and_pullback(
    f, backend::AutoStochasticAD, x, dy, extras::NoPullbackExtras
)
    return f(x), DI.pullback(f, backend, x, dy, extras)
end

end # module
