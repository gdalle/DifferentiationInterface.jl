using ADTypes: AutoChainRules
using DifferentiationInterface: CustomImplem, FallbackImplem
using Diffractor: DiffractorRuleConfig

test_pushforward(AutoChainRules(DiffractorRuleConfig()), scenarios; type_stability=false);
test_jacobian_and_friends(
    AutoChainRules(DiffractorRuleConfig()), scenarios, CustomImplem(); type_stability=false
);
test_jacobian_and_friends(
    AutoChainRules(DiffractorRuleConfig()),
    scenarios,
    FallbackImplem();
    type_stability=false,
);
