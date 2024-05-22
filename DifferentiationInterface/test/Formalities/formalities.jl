using Aqua: Aqua
using DifferentiationInterface
using Documenter: Documenter
using JET: JET
using JuliaFormatter: JuliaFormatter
using Test

@testset "Aqua" begin
    Aqua.test_all(
        DifferentiationInterface; ambiguities=false, deps_compat=(check_extras = false)
    )
end

@testset "JuliaFormatter" begin
    @test JuliaFormatter.format(DifferentiationInterface; verbose=false, overwrite=false)
end

@testset verbose = true "JET" begin
    if VERSION >= v"1.10"
        JET.test_package(DifferentiationInterface; target_defined_modules=true)
    end
end

Documenter.doctest(DifferentiationInterface)
