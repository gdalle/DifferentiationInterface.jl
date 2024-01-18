"""
    DifferentiationInterface

An experimental redesign for [AbstractDifferentiation.jl]
(https://github.com/JuliaDiff/AbstractDifferentiation.jl).

# Exports

$(EXPORTS)
"""
module DifferentiationInterface

using DocStringExtensions

abstract type AbstractBackend end

"""
    ChainRulesBackend
"""
struct ChainRulesBackend <: AbstractBackend end

"""
    EnzymeBackend
"""
struct EnzymeBackend <: AbstractBackend end

"""
    ForwardDiffBackend
"""
struct ForwardDiffBackend <: AbstractBackend end

"""
    jvp!(dy, backend, f, x, dx)

Compute a Jacobian-vector product and return `dy`.

# Arguments

- `dy`: cotangent, might be overwritten
- `backend::AbstractBackend`: autodiff backend
- `f`: function `x -> y` to differentiate
- `x`: argument
- `dx`: tangent
"""
function jvp! end

"""
    jvp!(dx, backend, f, x, dy)

Compute a Jacobian-vector product and return `dx`.

# Arguments

- `dx`: tangent, might be overwritten
- `backend::AbstractBackend`: autodiff backend
- `f`: function `x -> y` to differentiate
- `x`: argument
- `dy`: cotangent
"""
function vjp! end

export jvp!, vjp!
export ChainRulesBackend, EnzymeBackend, ForwardDiffBackend

end
