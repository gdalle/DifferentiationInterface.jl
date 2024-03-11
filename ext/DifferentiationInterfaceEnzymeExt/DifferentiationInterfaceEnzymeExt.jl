module DifferentiationInterfaceEnzymeExt

using ADTypes: AutoEnzyme
using DifferentiationInterface: CustomImplem
import DifferentiationInterface as DI
using DocStringExtensions
using Enzyme:
    Active,
    Duplicated,
    Forward,
    Reverse,
    ReverseWithPrimal,
    autodiff,
    gradient,
    gradient!,
    jacobian

# Enzyme's `Duplicated(x, dx)` expects both arguments to be of the same type
function DI.basisarray(::AutoEnzyme, a::AbstractArray{T}, i::CartesianIndex) where {T}
    b = zero(a)
    b[i] = one(T)
    return b
end

include("forward.jl")
include("reverse.jl")

end # module
