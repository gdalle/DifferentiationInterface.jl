using ADTypes: AutoReverseDiff
using DifferentiationInterface: CustomImplem, FallbackImplem
using ReverseDiff: ReverseDiff

test_pullback(AutoReverseDiff(), scenarios; type_stability=false);
test_jacobian_and_friends(
    AutoReverseDiff(), scenarios, CustomImplem(); type_stability=false
);
test_jacobian_and_friends(
    AutoReverseDiff(), scenarios, FallbackImplem(); type_stability=false
);
