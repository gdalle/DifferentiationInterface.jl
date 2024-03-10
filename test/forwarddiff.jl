using ADTypes: AutoForwardDiff
using DifferentiationInterface: CustomImplem, FallbackImplem
using ForwardDiff: ForwardDiff

test_pushforward(AutoForwardDiff(), scenarios; type_stability=true);
test_jacobian_and_friends(
    CustomImplem(), AutoForwardDiff(), scenarios; type_stability=false
);
test_jacobian_and_friends(
    FallbackImplem(), AutoForwardDiff(), scenarios; type_stability=false
);
