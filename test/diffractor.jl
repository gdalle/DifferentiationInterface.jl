using ADTypes: AutoDiffractor
using Diffractor: Diffractor
using DifferentiationInterface.DifferentiationTest

test_operators_allocating(AutoDiffractor(); type_stability=false);

test_second_order_operators_allocating(
    AutoDiffractor(); excluded=[:hessian], type_stability=false
)
