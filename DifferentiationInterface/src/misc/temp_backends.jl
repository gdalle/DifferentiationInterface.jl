"""
    AutoGTPSA{D}

Struct used to select the [GTPSA.jl](https://github.com/bmad-sim/GTPSA.jl) backend for automatic differentiation.

!!! warning
    This type is not part of the public API, it is only used for testing before being moved to ADTypes.jl.

# Constructors

    AutoGTPSA(; descriptor=nothing)

# Fields

  - `descriptor::D`: can be either

      + a GTPSA `Descriptor` specifying the number of variables/parameters, parameter 
        order, individual variable/parameter truncation orders, and maximum order. See 
        the [GTPSA.jl documentation](https://bmad-sim.github.io/GTPSA.jl/stable/man/c_descriptor/) for more details.
      + `nothing` to automatically use a `Descriptor` given the context.
      
"""
Base.@kwdef struct AutoGTPSA{D} <: AbstractADType
    descriptor::D = nothing
end

ADTypes.mode(::AutoGTPSA) = ForwardMode()
