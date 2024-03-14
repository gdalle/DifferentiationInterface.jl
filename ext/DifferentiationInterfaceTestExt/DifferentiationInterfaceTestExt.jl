module DifferentiationInterfaceTestExt

# package dependencies
using ADTypes: AbstractADType
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest
using DifferentiationInterface: ForwardMode, ReverseMode, autodiff_mode
import DifferentiationInterface as DI
import DifferentiationInterface.DifferentiationTest as DT
using LinearAlgebra: dot

# new dependencies
using ForwardDiff: ForwardDiff
using JET: @test_opt
using Random: AbstractRNG, default_rng, randn!
using Test: @test, @testset

include("scenarios.jl")
include("test_non_mutating.jl")
include("test_mutating.jl")

end
