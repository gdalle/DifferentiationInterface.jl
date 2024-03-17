using ADTypes: AutoZygote
using Zygote: Zygote
using DifferentiationInterface.DifferentiationTest

test_operators_allocating(AutoZygote(); type_stability=false);

test_second_order_operators_allocating(
    AutoZygote(); excluded=[:hessian], type_stability=false
)
