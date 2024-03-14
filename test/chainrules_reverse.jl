using ADTypes: AutoChainRules
using Zygote: ZygoteRuleConfig
using DifferentiationInterface.DifferentiationTest

test_all_operators(
    AutoChainRules(ZygoteRuleConfig()), default_scenarios(); type_stability=false
);
