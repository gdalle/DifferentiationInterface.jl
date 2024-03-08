using ADTypes: AutoEnzyme
using Enzyme: Enzyme

test_pullback(AutoEnzyme(Val(:reverse)), scenarios; type_stability=true);
test_jacobian_and_friends(AutoEnzyme(Val(:reverse)), scenarios; type_stability=true)
test_jacobian_and_friends(
    AutoEnzyme(Val(:reverse)), scenarios, Val(:fallback); type_stability=true
)
