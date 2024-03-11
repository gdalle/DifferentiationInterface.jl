using ADTypes: AutoEnzyme
using DifferentiationInterface: CustomImplem, FallbackImplem
using Enzyme: Enzyme

test_pushforward(AutoEnzyme(Val(:forward)), scenarios; type_stability=true);
test_jacobian_and_friends(
    CustomImplem(), AutoEnzyme(Val(:forward)), scenarios; type_stability=true
);
test_jacobian_and_friends(
    FallbackImplem(), AutoEnzyme(Val(:forward)), scenarios; type_stability=true
);
