using Pkg
Pkg.add(["ChainRulesCore", "Zygote"])

using ChainRulesCore
using DifferentiationInterface, DifferentiationInterfaceTest
using Test
using Zygote: ZygoteRuleConfig

LOGGING = get(ENV, "CI", "false") == "false"

for backend in [AutoChainRules(ZygoteRuleConfig())]
    @test check_available(backend)
    @test !check_inplace(backend)
end

test_differentiation(
    AutoChainRules(ZygoteRuleConfig()),
    default_scenarios();
    excluded=[:second_derivative],
    second_order=VERSION >= v"1.10",
    logging=LOGGING,
);

test_differentiation(
    AutoChainRules(ZygoteRuleConfig()),
    default_scenarios(; include_normal=false, include_constantified=true);
    second_order=false,
    logging=LOGGING,
);
