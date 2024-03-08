using ADTypes: AutoEnzyme
using Enzyme: Enzyme

test_pushforward(AutoEnzyme(Val(:forward)), scenarios; type_stability=true);
test_jacobian_and_friends(AutoEnzyme(Val(:forward)), scenarios; type_stability=true);
test_jacobian_and_friends(
    AutoEnzyme(Val(:forward)), scenarios, Val(:fallback); type_stability=true
);
