using DifferentiationInterface, DifferentiationInterfaceTest
using Zygote: Zygote

backends = [
    AutoChainRules(Zygote.ZygoteRuleConfig()), AutoZygote(), MyAutoSparse(AutoZygote())
]

for backend in backends
    @test check_available(backend)
    @test !check_twoarg(backend)
    if backend isa AutoZygote
        @test check_hessian(backend)
    end
end

test_differentiation(
    AutoChainRules(Zygote.ZygoteRuleConfig()); second_order=false, logging=LOGGING
);

test_differentiation(
    [AutoZygote(), MyAutoSparse(AutoZygote())];
    excluded=[SecondDerivativeScenario, HVPScenario],
    logging=LOGGING,
);

if VERSION >= v"1.10"
    test_differentiation(
        AutoZygote(),
        vcat(component_scenarios(), gpu_scenarios(), static_scenarios());
        second_order=false,
        logging=LOGGING,
    )
end
