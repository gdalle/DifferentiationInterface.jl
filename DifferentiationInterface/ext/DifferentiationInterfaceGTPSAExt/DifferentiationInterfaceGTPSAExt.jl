module DifferentiationInterfaceGTPSAExt

import DifferentiationInterface as DI
using DifferentiationInterface: AutoGTPSA  # TODO: replace with ADTypes
using DifferentiationInterface:
    DerivativeExtras,
    SecondDerivativeExtras,
    GradientExtras,
    JacobianExtras,
    HessianExtras,
    PushforwardExtras,
    HVPExtras,
    Tangents
using GTPSA

DI.check_available(::AutoGTPSA) = true

include("onearg.jl")
include("twoarg.jl")

end
