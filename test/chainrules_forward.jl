using ADTypes: AutoChainRules
using Diffractor: DiffractorRuleConfig

test_pushforward(AutoChainRules(DiffractorRuleConfig()), scenarios; type_stability=false);
test_jacobian_and_friends(
    AutoChainRules(DiffractorRuleConfig()), scenarios; type_stability=false
);
