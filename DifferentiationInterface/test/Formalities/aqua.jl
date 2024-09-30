using Aqua: Aqua
using DifferentiationInterface
using Test
using SparseMatrixColorings

Aqua.test_all(
    DifferentiationInterface; ambiguities=false, deps_compat=(check_extras = false)
)
