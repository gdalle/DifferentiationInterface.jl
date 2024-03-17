using ADTypes: AutoFiniteDiff
using FiniteDiff: FiniteDiff
using DifferentiationInterface.DifferentiationTest

test_operators(
    AutoFiniteDiff(); second_order=false, mutating=false, excluded=[:jacobian_allocating]
);
test_operators(AutoFiniteDiff(); operators=[:jacobian_allocating], type_stability=false);
