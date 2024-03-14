using ADTypes: AutoChainRules
using Diffractor: DiffractorRuleConfig
using DifferentiationInterface.DifferentiationTest

test_all_operators(
    AutoChainRules(DiffractorRuleConfig()), default_scenarios(); type_stability=false
);
