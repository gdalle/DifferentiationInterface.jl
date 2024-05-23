using Aqua: Aqua
using DifferentiationInterface
using Test

Aqua.test_all(
    DifferentiationInterface; ambiguities=false, deps_compat=(check_extras = false)
)
