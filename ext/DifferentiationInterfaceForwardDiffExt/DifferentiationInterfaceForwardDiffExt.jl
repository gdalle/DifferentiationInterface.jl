module DifferentiationInterfaceForwardDiffExt

using ADTypes: AbstractADType, AutoForwardDiff
using DifferentiationInterface:
    ForwardMode,
    ReverseMode,
    MutationSupported,
    MutationNotSupported,
    mode,
    mutation_behavior,
    outer
import DifferentiationInterface as DI
using DifferentiationInterface.DifferentiationTest
import DifferentiationInterface.DifferentiationTest as DT
using DiffResults: DiffResults
using DocStringExtensions
using ForwardDiff:
    Chunk,
    Dual,
    DerivativeConfig,
    ForwardDiff,
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
using LinearAlgebra: dot, mul!
using Test: @testset, @test

choose_chunk(::AutoForwardDiff{nothing}, x) = Chunk(x)
choose_chunk(::AutoForwardDiff{C}, x) where {C} = Chunk{C}()

tag_type(::F, ::V) where {F,V<:Number} = Tag{F,V}
tag_type(::F, ::AbstractArray{V}) where {F,V<:Number} = Tag{F,V}

include("allocating.jl")
include("mutating.jl")

include("test_correctness.jl")

end # module
