module DifferentiationInterfaceGTPSAExt

using ADTypes: AbstractADType, ForwardMode
import ADTypes: mode
import DifferentiationInterface as DI
using DifferentiationInterface:
    DerivativeExtras,
    SecondDerivativeExtras,
    GradientExtras,
    JacobianExtras,
    HessianExtras,
    PushforwardExtras
using GTPSA


"""
    AutoGTPSA{D}

Struct used to select the [GTPSA.jl](https://github.com/bmad-sim/GTPSA.jl) backend for automatic differentiation.

# Constructors

    AutoGTPSA(; descriptor=nothing)

# Fields

  - `descriptor::D`: can be either

      + a GTPSA `Descriptor` specifying the number of variables/parameters, parameter 
        order, individual variable/parameter truncation orders, and maximum order. See 
        [here](https://bmad-sim.github.io/GTPSA.jl/stable/man/c_descriptor/) for more details.
      + `nothing` to automatically use a `Descriptor` given the context
      
"""
Base.@kwdef struct AutoGTPSA{D} <: AbstractADType 
  descriptor::D = nothing
end

mode(::AutoGTPSA) = ForwardMode()

DI.check_available(::AutoGTPSA) = true

include("onearg.jl")

end # module
