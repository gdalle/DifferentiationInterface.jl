using ADTypes: AutoFiniteDiff
using DifferentiationInterface: CustomImplem, FallbackImplem
using FiniteDiff: FiniteDiff

test_pushforward(AutoFiniteDiff(), scenarios; type_stability=true);
test_jacobian_and_friends(AutoFiniteDiff(), scenarios, CustomImplem(); type_stability=false);
test_jacobian_and_friends(
    AutoFiniteDiff(), scenarios, FallbackImplem(); type_stability=false
);
