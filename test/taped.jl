using Pkg
Pkg.add(; url="https://github.com/withbayes/Taped.jl/")

include("test_imports.jl")

using DifferentiationInterface: AutoTaped
using Taped: Taped

@test check_available(AutoTaped())
@test !check_mutation(AutoTaped())
@test !check_hessian(AutoTaped())

test_differentiation(AutoTaped(); second_order=false, excluded=[jacobian]);
