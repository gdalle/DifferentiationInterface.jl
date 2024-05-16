import DifferentiationInterfaceTest as DIT
using Test

A = [
    1 0 0
    0 1 0
    0 1 0
    0 0 0
]

@testset "Column coloring" begin
    A = [
        1 0 0
        0 1 0
        0 1 1
    ]
    @test DIT.check_structurally_orthogonal_columns(A, [1, 2, 3])
    @test DIT.check_structurally_orthogonal_columns(A, [1, 2, 1])
    @test DIT.check_structurally_orthogonal_columns(A, [1, 1, 2])
    @test !DIT.check_structurally_orthogonal_columns(A, [1, 2, 2])
end

@testset "Row coloring" begin
    A = [
        1 0 0
        0 1 0
        0 1 1
    ]
    @test DIT.check_structurally_orthogonal_rows(A, [1, 2, 3])
    @test DIT.check_structurally_orthogonal_rows(A, [1, 2, 1])
    @test DIT.check_structurally_orthogonal_rows(A, [1, 1, 2])
    @test !DIT.check_structurally_orthogonal_rows(A, [1, 2, 2])
end

@testset "Symmetric coloring" begin
    # example from "What color is your Jacobian", fig 4.1
    A = [
        1 1 0 0 0 0
        1 1 1 0 1 1
        0 1 1 1 0 0
        0 0 1 1 0 1
        0 1 0 0 1 0
        0 1 0 1 0 1
    ]
    @test DIT.check_symmetrically_structurally_orthogonal(A, [1, 2, 1, 3, 1, 1])
    @test !DIT.check_symmetrically_structurally_orthogonal(A, [1, 3, 1, 3, 1, 1])
end
