using Aqua: Aqua
using DifferentiationInterface
using JET: JET
using JuliaFormatter: JuliaFormatter
using Test
using SparseMatrixColorings

@testset verbose = true "Formalities" begin
    @testset "Aqua" begin
        Aqua.test_all(
            DifferentiationInterface; ambiguities=false, deps_compat=(check_extras = false)
        )
    end
    @testset "JET" begin
        JET.test_package(DifferentiationInterface; target_defined_modules=true)
    end
    @testset "JuliaFormatter" begin
        @test JuliaFormatter.format(
            DifferentiationInterface; verbose=false, overwrite=false
        )
    end
end
