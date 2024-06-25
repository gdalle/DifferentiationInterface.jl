module DifferentiationInterfaceReactantExt

using ADTypes: ADTypes
using Compat
import DifferentiationInterface as DI
using DifferentiationInterface:
    ReactantBackend,
    DerivativeExtras,
    GradientExtras,
    HessianExtras,
    HVPExtras,
    JacobianExtras,
    PullbackExtras,
    PushforwardExtras,
    SecondDerivativeExtras
using Reactant: ConcreteRArray, compile

DI.check_available(rebackend::ReactantBackend) = DI.check_available(rebackend.backend)
DI.mode(rebackend::ReactantBackend) = DI.mode(rebackend.backend)
DI.twoarg_support(rebackend::ReactantBackend) = DI.twoarg_support(rebackend.backend)

include("onearg.jl")

end # module
