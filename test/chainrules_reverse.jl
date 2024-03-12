using ADTypes: AutoChainRules
using DifferentiationInterface: CustomImplem, FallbackImplem
using Zygote: ZygoteRuleConfig

test_pullback(AutoChainRules(ZygoteRuleConfig()), scenarios; type_stability=false);
test_jacobian_and_friends(
    CustomImplem(), AutoChainRules(ZygoteRuleConfig()), scenarios; type_stability=false
);
test_jacobian_and_friends(
    FallbackImplem(), AutoChainRules(ZygoteRuleConfig()), scenarios; type_stability=false
);
