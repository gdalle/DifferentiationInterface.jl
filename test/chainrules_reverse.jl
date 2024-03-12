using ADTypes: AutoChainRules
using Zygote: ZygoteRuleConfig

test_pullback(AutoChainRules(ZygoteRuleConfig()), scenarios; type_stability=false);
test_jacobian_and_friends(
    AutoChainRules(ZygoteRuleConfig()), scenarios; type_stability=false
);
