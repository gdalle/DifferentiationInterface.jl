using ADTypes: AutoForwardDiff
using DifferentiationInterface: CustomImplem, FallbackImplem
using ForwardDiff: ForwardDiff

test_pushforward(AutoForwardDiff(), scenarios; type_stability=true);
test_jacobian_and_friends(
    AutoForwardDiff(), scenarios, CustomImplem(); type_stability=false
);
test_jacobian_and_friends(
    AutoForwardDiff(), scenarios, FallbackImplem(); type_stability=false
);
