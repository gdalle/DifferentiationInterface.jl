alg = DI.GreedyColoringAlgorithm()

A = sprand(Bool, 100, 200, 0.05)

column_colors = ADTypes.column_coloring(A, alg)
@test DI.check_structurally_orthogonal_columns(A, column_colors)
@test maximum(column_colors) < size(A, 2) ÷ 2

row_colors = ADTypes.row_coloring(A, alg)
@test DI.check_structurally_orthogonal_rows(A, row_colors)
@test maximum(row_colors) < size(A, 1) ÷ 2

S = Symmetric(sprand(Bool, 100, 100, 0.05)) + I
symmetric_colors = ADTypes.symmetric_coloring(S, alg)
@test DI.check_symmetrically_structurally_orthogonal(S, symmetric_colors)
@test maximum(symmetric_colors) < size(A, 2) ÷ 2
