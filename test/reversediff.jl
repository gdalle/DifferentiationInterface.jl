using ADTypes: AutoReverseDiff
using DifferentiationInterface: CustomImplem, FallbackImplem
using ReverseDiff: ReverseDiff

test_pullback(AutoReverseDiff(), scenarios; type_stability=false);
test_jacobian_and_friends(
    CustomImplem(), AutoReverseDiff(), scenarios; type_stability=false
);
test_jacobian_and_friends(
    FallbackImplem(), AutoReverseDiff(), scenarios; type_stability=false
);
