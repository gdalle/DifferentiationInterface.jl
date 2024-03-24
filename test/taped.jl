using Pkg
Pkg.add(; url="https://github.com/withbayes/Taped.jl/")

using DifferentiationInterface
using DifferentiationInterface: AutoTaped
using DifferentiationInterface.DifferentiationTest
using Taped: Taped

using ForwardDiff: ForwardDiff
using JET: JET
using Test

@test check_available(AutoTaped())
@test !check_mutation(AutoTaped())
@test !check_hessian(AutoTaped())

test_operators(AutoTaped(); second_order=false, type_stability=false, excluded=[jacobian]);
