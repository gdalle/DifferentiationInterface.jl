module DifferentiationInterfaceReactantExt

using ADTypes: ADTypes
using Compat
import DifferentiationInterface as DI
using DifferentiationInterface:
     GradientExtras
using Reactant: ConcreteRArray, compile

struct ReactantBackend{B} <: ADTypes.AbstractADType
    backend::B
end

DI.check_available(rebackend::ReactantBackend) = DI.check_available(rebackend.backend)
DI.mode(rebackend::ReactantBackend) = DI.mode(rebackend.backend)
DI.twoarg_support(rebackend::ReactantBackend) = DI.twoarg_support(rebackend.backend)

include("onearg.jl")

end # module
