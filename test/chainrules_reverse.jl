using ADTypes: AutoChainRules
using DifferentiationInterface: CustomImplem, FallbackImplem
using Zygote: ZygoteRuleConfig

test_pullback(AutoChainRules(ZygoteRuleConfig()), scenarios; type_stability=false);
test_jacobian_and_friends(
    AutoChainRules(ZygoteRuleConfig()), scenarios, CustomImplem(); type_stability=false
);
test_jacobian_and_friends(
    AutoChainRules(ZygoteRuleConfig()), scenarios, FallbackImplem(); type_stability=false
);
