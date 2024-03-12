using ADTypes: AutoDiffractor
using DifferentiationInterface: CustomImplem, FallbackImplem
using Diffractor: Diffractor

test_pushforward(AutoDiffractor(), scenarios; type_stability=false);
test_jacobian_and_friends(AutoDiffractor(), scenarios, CustomImplem(); type_stability=false);
test_jacobian_and_friends(
    AutoDiffractor(), scenarios, FallbackImplem(); type_stability=false
);
