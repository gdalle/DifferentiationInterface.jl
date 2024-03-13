module DifferentiationInterfaceEnzymeExt

using ADTypes: AutoEnzyme
import DifferentiationInterface as DI
using DocStringExtensions
using Enzyme:
    Active,
    Const,
    Duplicated,
    Forward,
    Reverse,
    ReverseWithPrimal,
    autodiff,
    gradient,
    gradient!,
    jacobian

"""
    AutoEnzyme(Val(:forward))
    AutoEnzyme(Val(:reverse))

Construct a forward or reverse mode `AutoEnzyme` backend.

!!! warning
    This is the mode convention chosen by DifferentiationInterface.jl, for lack of a global consensus (see [ADTypes.jl#24](https://github.com/SciML/ADTypes.jl/issues/24)).
"""
AutoEnzyme

# Enzyme's `Duplicated(x, dx)` expects both arguments to be of the same type
function DI.basisarray(::AutoEnzyme, a::AbstractArray{T}, i::CartesianIndex) where {T}
    b = zero(a)
    b[i] = one(T)
    return b
end

include("forward.jl")
include("reverse.jl")

end # module
