using Aqua: Aqua
using DifferentiationInterface
using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using JET: JET
using JuliaFormatter: JuliaFormatter
using Test
using Zygote: Zygote

@testset verbose = true "DifferentiationInterface.jl" begin
    @testset "Aqua" begin
        Aqua.test_all(DifferentiationInterface; deps_compat=(; check_extras=false,))
    end
    @testset "JuliaFormatter" begin
        @test JuliaFormatter.format(
            DifferentiationInterface; verbose=false, overwrite=false
        )
    end
    @testset "JET" begin
        JET.test_package(DifferentiationInterface; target_defined_modules=true)
    end
    @testset "ForwardDiff" begin
        include("forwarddiff.jl")
    end
end
