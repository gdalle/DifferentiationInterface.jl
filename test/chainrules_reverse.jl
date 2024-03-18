using ADTypes: AutoChainRules
using Zygote: ZygoteRuleConfig
using DifferentiationInterface.DifferentiationTest

test_operators(AutoChainRules(ZygoteRuleConfig()); second_order=false, type_stability=false);
