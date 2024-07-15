module DifferentiationInterfaceGTPSAExt

import DifferentiationInterface as DI
using DifferentiationInterface: AutoGTPSA  # TODO: replace with ADTypes
using DifferentiationInterface:
    DerivativeExtras,
    SecondDerivativeExtras,
    GradientExtras,
    JacobianExtras,
    HessianExtras,
    PushforwardExtras
using GTPSA

DI.check_available(::AutoGTPSA) = true
DI.twoarg_support(::AutoGTPSA) = DI.TwoArgNotSupported()

include("onearg.jl")

end # module
