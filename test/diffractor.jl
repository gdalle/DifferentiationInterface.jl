using DifferentiationInterface: AutoDiffractor, CustomImplem, FallbackImplem
using Diffractor: Diffractor

test_pushforward(AutoDiffractor(), scenarios; type_stability=true);
test_jacobian_and_friends(CustomImplem(), AutoDiffractor(), scenarios; type_stability=true);
test_jacobian_and_friends(
    FallbackImplem(), AutoDiffractor(), scenarios; type_stability=true
);
