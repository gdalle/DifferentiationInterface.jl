using ADTypes: AutoDiffractor
using Diffractor: Diffractor
using DifferentiationInterface.DifferentiationTest

test_operators(
    AutoDiffractor(); mutating=false, excluded=[:hessian_allocating], type_stability=false
);
