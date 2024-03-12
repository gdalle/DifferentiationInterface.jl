using ADTypes: AutoEnzyme
using DifferentiationInterface: CustomImplem, FallbackImplem
using Enzyme: Enzyme

test_pushforward(AutoEnzyme(Val(:forward)), scenarios; type_stability=true);
test_jacobian_and_friends(
    AutoEnzyme(Val(:forward)), scenarios, CustomImplem(); type_stability=true
);
test_jacobian_and_friends(
    AutoEnzyme(Val(:forward)), scenarios, FallbackImplem(); type_stability=true
);
