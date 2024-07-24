module DifferentiationInterfaceStochasticADExt

import DifferentiationInterface as DI
using DifferentiationInterface: AutoStochasticAD
using DifferentiationInterface: NoPushforwardExtras, NoPullbackExtras
using StochasticAD: derivative_estimate

DI.check_available(::AutoStochasticAD) = true
DI.twoarg_support(::AutoStochasticAD) = DI.TwoArgNotSupported()

DI.prepare_pushforward(f, ::AutoStochasticAD, x, dx) = NoPushforwardExtras()

function DI.pushforward(
    f, ad::AutoStochasticAD, x, dx, ::NoPushforwardExtras)
    return sum(derivative_estimate(f, x, direction=dx) for _ in 1:ad.n_samples) / ad.n_samples
end

function DI.value_and_pushforward(
    f, backend::AutoStochasticAD, x, dx, extras::NoPushforwardExtras
)
    return f(x), DI.pushforward(f, backend, x, dx, extras)
end

end # module
