using ADTypes: AutoZygote
using DifferentiationInterface: CustomImplem, FallbackImplem
using Zygote: Zygote

test_pullback(AutoZygote(), scenarios; type_stability=false);
test_jacobian_and_friends(CustomImplem(), AutoZygote(), scenarios; type_stability=false);
test_jacobian_and_friends(FallbackImplem(), AutoZygote(), scenarios; type_stability=false);
