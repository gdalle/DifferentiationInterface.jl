using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff

test_pushforward(AutoForwardDiff(), scenarios; type_stability=true);
test_jacobian_and_friends(AutoForwardDiff(), scenarios; type_stability=false);
test_jacobian_and_friends(
    AutoForwardDiff(), scenarios, Val(:fallback); type_stability=false
);
