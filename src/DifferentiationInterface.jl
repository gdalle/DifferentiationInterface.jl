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

Performs autodiff with any package based on [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl).
"""
struct ChainRulesBackend <: AbstractBackend end

"""
    EnzymeBackend

Performs autodiff with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl).
"""
struct EnzymeBackend <: AbstractBackend end

"""
    ForwardDiffBackend

Performs autodiff with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl).
"""
struct ForwardDiffBackend <: AbstractBackend end

"""
    jvp!(dy, backend, f, x, dx)

Compute a Jacobian-vector product and return `dy`.

# Arguments

- `dy`: cotangent, might be overwritten
- `backend`: autodiff backend
- `f`: function `x -> y` to differentiate
- `x`: argument
- `dx`: tangent
"""
function jvp! end

"""
    vjp!(dx, backend, f, x, dy)

Compute a vector-Jacobian product and return `dx`.

# Arguments

- `dx`: tangent, might be overwritten
- `backend`: autodiff backend
- `f`: function `x -> y` to differentiate
- `x`: argument
- `dy`: cotangent
"""
function vjp! end

export jvp!, vjp!
export ChainRulesBackend, EnzymeBackend, ForwardDiffBackend

end
