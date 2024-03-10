using ADTypes: AutoEnzyme
using DifferentiationInterface: CustomImplem, FallbackImplem
using Enzyme: Enzyme

test_pullback(AutoEnzyme(Val(:reverse)), scenarios; type_stability=true);
test_jacobian_and_friends(
    CustomImplem(), AutoEnzyme(Val(:reverse)), scenarios; type_stability=true
)
test_jacobian_and_friends(
    FallbackImplem(), AutoEnzyme(Val(:reverse)), scenarios; type_stability=true
)
