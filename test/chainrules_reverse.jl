include("test_imports.jl")

using Zygote: ZygoteRuleConfig

@test check_available(AutoChainRules(ZygoteRuleConfig()))
@test !check_mutation(AutoChainRules(ZygoteRuleConfig()))
@test_broken !check_hessian(AutoChainRules(ZygoteRuleConfig()))

test_differentiation(AutoChainRules(ZygoteRuleConfig()); second_order=false);
