using ADTypes: AutoChainRules
using Zygote: ZygoteRuleConfig
using DifferentiationInterface.DifferentiationTest

test_operators(
    AutoChainRules(ZygoteRuleConfig());
    mutating=false,
    excluded=[:hessian_allocating],
    type_stability=false,
);
