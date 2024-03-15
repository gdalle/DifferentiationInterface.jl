using ADTypes: AutoFiniteDiff
using FiniteDiff: FiniteDiff
using DifferentiationInterface.DifferentiationTest

test_operators_allocating(AutoFiniteDiff(); excluded=[:jacobian]);
test_operators_allocating(AutoFiniteDiff(); included=[:jacobian], type_stability=false);
