using ADTypes: ADTypes
import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using LinearAlgebra: I, Symmetric
using SparseArrays: sprand
using Test

alg = DI.GreedyColoringAlgorithm()

@testset "Grouping" begin
    colors = [1, 3, 1, 3, 1, 2]
    @test DI.color_groups(colors) == [[1, 3, 5], [6], [2, 4]]
    colors = [2, 3, 2, 3, 2, 1]
    @test DI.color_groups(colors) == [[6], [1, 3, 5], [2, 4]]
    colors = [2, 3, 2, 3, 2]
    @test_throws AssertionError DI.color_groups(colors)
end

@testset "Column coloring" begin
    for A in (sprand(Bool, 100, 200, 0.05), sprand(Bool, 200, 100, 0.05))
        column_colors = ADTypes.column_coloring(A, alg)
        @test DIT.check_structurally_orthogonal_columns(A, column_colors)
        @test minimum(column_colors) == 1
        @test maximum(column_colors) < size(A, 2) ÷ 2
    end
end

@testset "Row coloring" begin
    for A in (sprand(Bool, 100, 200, 0.05), sprand(Bool, 200, 100, 0.05))
        row_colors = ADTypes.row_coloring(A, alg)
        @test DIT.check_structurally_orthogonal_rows(A, row_colors)
        @test minimum(row_colors) == 1
        @test maximum(row_colors) < size(A, 1) ÷ 2
    end
end

@testset "Symmetric coloring" begin
    S = Symmetric(sprand(Bool, 100, 100, 0.05)) + I
    symmetric_colors = ADTypes.symmetric_coloring(S, alg)
    @test DIT.check_symmetrically_structurally_orthogonal(S, symmetric_colors)
    @test minimum(symmetric_colors) == 1
    @test maximum(symmetric_colors) < size(S, 2) ÷ 2
end
