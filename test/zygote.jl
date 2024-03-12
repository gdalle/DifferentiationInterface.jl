using ADTypes: AutoZygote
using DifferentiationInterface: CustomImplem, FallbackImplem
using Zygote: Zygote

test_pullback(AutoZygote(), scenarios; type_stability=false);
test_jacobian_and_friends(AutoZygote(), scenarios, CustomImplem(); type_stability=false);
test_jacobian_and_friends(AutoZygote(), scenarios, FallbackImplem(); type_stability=false);
