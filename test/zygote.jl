using ADTypes: AutoZygote
using Zygote: Zygote
using DifferentiationInterface.DifferentiationTest

test_operators(
    AutoZygote(); mutating=false, excluded=[:hessian_allocating], type_stability=false
);
