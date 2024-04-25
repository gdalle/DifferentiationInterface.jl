using ADTypes: column_coloring, row_coloring, symmetric_coloring
using DifferentiationInterface:
    GreedyColoringAlgorithm,
    check_structurally_orthogonal_columns,
    check_structurally_orthogonal_rows,
    check_symmetrically_structurally_orthogonal
using LinearAlgebra
using SparseArrays
using Test

alg = GreedyColoringAlgorithm()

A = sprand(Bool, 100, 200, 0.1)

column_colors = column_coloring(A, alg)
@test check_structurally_orthogonal_columns(A, column_colors)
@test maximum(column_colors) < size(A, 2) ÷ 2

row_colors = row_coloring(A, alg)
@test check_structurally_orthogonal_rows(A, row_colors)
@test maximum(row_colors) < size(A, 1) ÷ 2

S = Symmetric(sprand(Bool, 100, 100, 0.1)) + I
symmetric_colors = symmetric_coloring(S, alg)
@test check_symmetrically_structurally_orthogonal(S, symmetric_colors)
@test maximum(symmetric_colors) < size(A, 2) ÷ 2
