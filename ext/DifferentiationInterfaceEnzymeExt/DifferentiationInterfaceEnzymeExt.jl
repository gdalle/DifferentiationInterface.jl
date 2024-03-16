module DifferentiationInterfaceEnzymeExt

using ADTypes: AutoEnzyme
import DifferentiationInterface as DI
using DocStringExtensions
using Enzyme:
    Active,
    Const,
    Duplicated,
    DuplicatedNoNeed,
    Forward,
    ForwardMode,
    Reverse,
    ReverseWithPrimal,
    ReverseMode,
    autodiff,
    gradient,
    gradient!,
    jacobian

"""
    AutoEnzyme(Enzyme.Forward)
    AutoEnzyme(Enzyme.Reverse)

Construct a forward or reverse mode `AutoEnzyme` backend.
"""
AutoEnzyme

const AutoForwardEnzyme = AutoEnzyme{<:ForwardMode}
const AutoReverseEnzyme = AutoEnzyme{<:ReverseMode}

function DI.mode(::AutoEnzyme)
    return error(
        "You need to specify the Enzyme mode with `AutoEnzyme(Enzyme.Forward)` or `AutoEnzyme(Enzyme.Reverse)`",
    )
end

DI.mode(::AutoForwardEnzyme) = DI.ForwardMode()
DI.mode(::AutoReverseEnzyme) = DI.ReverseMode()

# Enzyme's `Duplicated(x, dx)` expects both arguments to be of the same type
function DI.basisarray(::AutoEnzyme, a::AbstractArray{T}, i::CartesianIndex) where {T}
    b = zero(a)
    b[i] = one(T)
    return b
end

include("forward_allocating.jl")
include("forward_mutating.jl")

include("reverse_allocating.jl")
include("reverse_mutating.jl")

end # module
