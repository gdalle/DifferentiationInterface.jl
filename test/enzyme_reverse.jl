using ADTypes: AutoEnzyme
using DifferentiationInterface: CustomImplem, FallbackImplem
using Enzyme: Enzyme

test_pullback(AutoEnzyme(Val(:reverse)), scenarios; type_stability=true);
test_jacobian_and_friends(
    AutoEnzyme(Val(:reverse)), scenarios, CustomImplem(); type_stability=true
)
test_jacobian_and_friends(
    AutoEnzyme(Val(:reverse)), scenarios, FallbackImplem(); type_stability=true
)
