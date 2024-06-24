module DifferentiationInterfaceGTPSAExt

using ADTypes: AutoGTPSA
import DifferentiationInterface as DI
using DifferentiationInterface:
    DerivativeExtras,
    SecondDerivativeExtras,
    GradientExtras,
    JacobianExtras,
    HessianExtras,
    PushforwardExtras,
    NoPushforwardExtras
using GTPSA

DI.check_available(::AutoGTPSA) = true

include("onearg.jl")

end # module
