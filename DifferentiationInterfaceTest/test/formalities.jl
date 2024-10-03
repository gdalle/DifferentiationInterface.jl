using DifferentiationInterface
using DifferentiationInterfaceTest
using Aqua: Aqua
using JET: JET
using JuliaFormatter: JuliaFormatter
using SparseMatrixColorings: SparseMatrixColorings
using Test

@testset "Aqua" begin
    Aqua.test_all(
        DifferentiationInterfaceTest; ambiguities=false, deps_compat=(check_extras = false)
    )
end
@testset "JuliaFormatter" begin
    @test JuliaFormatter.format(
        DifferentiationInterfaceTest; verbose=false, overwrite=false
    )
end
@testset verbose = true "JET" begin
    JET.test_package(DifferentiationInterfaceTest; target_defined_modules=true)
end
