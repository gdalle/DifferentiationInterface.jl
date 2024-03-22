module DifferentiationInterfaceForwardDiffExt

using ADTypes:
    AbstractADType,
    AbstractFiniteDifferencesMode,
    AbstractForwardMode,
    AbstractReverseMode,
    AbstractSymbolicDifferentiationMode,
    AutoForwardDiff
using DifferentiationInterface:
    inner,
    mode,
    outer,
    supports_mutation,
    supports_pushforward,
    supports_pullback,
    supports_hvp
import DifferentiationInterface as DI
using DifferentiationInterface.DifferentiationTest
import DifferentiationInterface.DifferentiationTest as DT
using DifferentiationInterface.DifferentiationTest:
    AbstractOperator,
    PushforwardAllocating,
    PushforwardMutating,
    PullbackAllocating,
    PullbackMutating,
    MultiderivativeAllocating,
    MultiderivativeMutating,
    GradientAllocating,
    JacobianAllocating,
    JacobianMutating,
    DerivativeAllocating,
    SecondDerivativeAllocating,
    HessianAllocating,
    HessianVectorProductAllocating,
    compatible_scenarios
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
