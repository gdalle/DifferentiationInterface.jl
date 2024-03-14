module DifferentiationInterfaceForwardDiffExt

using ADTypes: AutoForwardDiff
import DifferentiationInterface as DI
using DiffResults: DiffResults
using DocStringExtensions
using ForwardDiff:
    Chunk,
    Dual,
    DerivativeConfig,
    GradientConfig,
    JacobianConfig,
    Tag,
    derivative,
    derivative!,
    extract_derivative,
    extract_derivative!,
    gradient,
    gradient!,
    jacobian,
    jacobian!,
    value
using LinearAlgebra: mul!

choose_chunk(::AutoForwardDiff{nothing}, x) = Chunk(x)
choose_chunk(::AutoForwardDiff{C}, x) where {C} = Chunk{C}()

tag_type(::F, ::V) where {F,V<:Number} = Tag{F,V}
tag_type(::F, ::AbstractArray{V}) where {F,V<:Number} = Tag{F,V}

include("non_mutating.jl")
include("mutating.jl")

end # module
