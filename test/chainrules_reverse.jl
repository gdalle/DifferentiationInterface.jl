using ADTypes: AutoChainRules
using Zygote: ZygoteRuleConfig
using DifferentiationInterface.DifferentiationTest

test_operators_allocating(AutoChainRules(ZygoteRuleConfig()); type_stability=false);

test_second_order_operators_allocating(
    AutoChainRules(ZygoteRuleConfig()); excluded=[:hessian], type_stability=false
)
