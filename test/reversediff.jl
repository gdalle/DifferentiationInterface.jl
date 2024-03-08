using ADTypes: AutoReverseDiff
using ReverseDiff: ReverseDiff

test_pullback(AutoReverseDiff(), scenarios; type_stability=false);
test_jacobian_and_friends(AutoReverseDiff(), scenarios; type_stability=false);
test_jacobian_and_friends(
    AutoReverseDiff(), scenarios, Val(:fallback); type_stability=false
);
