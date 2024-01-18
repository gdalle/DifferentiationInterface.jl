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
    jvp!(dys, backend, f, x, dx)

Compute a Jacobian-vector product and return `dys`.

# Arguments

- `dys::Tuple`: function argument cotangents, might be overwritten
- `backend::AbstractBackend`: autodiff backend to use
- `f`: function to differentiate
- `x::Tuple`: function arguments
- `dx::Tuple`: function argument tangents
"""
function jvp! end

"""
    jvp!(dxs, backend, f, x, dy)

Compute a Jacobian-vector product and return `dxs`.

# Arguments

- `dxs::Tuple`: function argument tangents, might be overwritten
- `backend::AbstractBackend`: autodiff backend to use
- `f`: function to differentiate
- `x::Tuple`: function arguments
- `dy::Tuple`: function argument cotangents
"""
function vjp! end

export jvp!, vjp!
export ChainRulesBackend, EnzymeBackend, ForwardDiffBackend

end
