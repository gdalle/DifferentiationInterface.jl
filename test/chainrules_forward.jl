using ADTypes: AutoChainRules
using Diffractor: DiffractorRuleConfig
using DifferentiationInterface.DifferentiationTest

test_operators_allocating(AutoChainRules(DiffractorRuleConfig()); type_stability=false);
