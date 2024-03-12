using ADTypes: AutoChainRules
using DifferentiationInterface: CustomImplem, FallbackImplem
using Diffractor: DiffractorRuleConfig

test_pushforward(AutoChainRules(DiffractorRuleConfig()), scenarios; type_stability=false);
test_jacobian_and_friends(
    CustomImplem(), AutoChainRules(DiffractorRuleConfig()), scenarios; type_stability=false
);
test_jacobian_and_friends(
    FallbackImplem().AutoChainRules(DiffractorRuleConfig()), scenarios; type_stability=false
);
