module DifferentiationInterfaceTestExt

# package dependencies
using ADTypes: AbstractADType
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using DifferentiationInterface: ForwardMode, ReverseMode, autodiff_mode
import DifferentiationInterface as DI
import DifferentiationInterface.DifferentiationTest as DT
using DocStringExtensions
using LinearAlgebra: dot

# new dependencies
using ForwardDiff: ForwardDiff
using JET: @test_opt
using Test: @test, @testset

include("scenarios.jl")
include("correctness.jl")
include("type_stability.jl")
include("test_allocating.jl")
include("test_mutating.jl")

end
