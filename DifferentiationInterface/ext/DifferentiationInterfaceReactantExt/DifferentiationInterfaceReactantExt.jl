module DifferentiationInterfaceReactantExt

using ADTypes: ADTypes
using Compat
import DifferentiationInterface as DI
using DifferentiationInterface:
    ReactantBackend,
    DerivativePrep,
    GradientPrep,
    HessianPrep,
    HVPPrep,
    JacobianPrep,
    PullbackPrep,
    PushforwardPrep,
    SecondDerivativePrep
using Reactant: ConcreteRArray, compile

DI.check_available(rebackend::ReactantBackend) = DI.check_available(rebackend.backend)
DI.mode(rebackend::ReactantBackend) = DI.mode(rebackend.backend)
DI.inplace_support(rebackend::ReactantBackend) = DI.inplace_support(rebackend.backend)

include("onearg.jl")

end # module
