using ADTypes: AutoFiniteDiff
using DifferentiationInterface: CustomImplem, FallbackImplem
using FiniteDiff: FiniteDiff

test_pushforward(AutoFiniteDiff(), scenarios; type_stability=true);
test_jacobian_and_friends(CustomImplem(), AutoFiniteDiff(), scenarios; type_stability=false);
test_jacobian_and_friends(
    FallbackImplem(), AutoFiniteDiff(), scenarios; type_stability=false
);
