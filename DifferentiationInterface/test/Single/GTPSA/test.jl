using DifferentiationInterface, DifferentiationInterfaceTest
using GTPSA: GTPSA
using Test

for backend in [AutoGTPSA()]
    @test check_available(backend)
    #@test check_twoarg(backend)
    #@test check_hessian(backend)
end

test_differentiation(
    AutoGTPSA(); excluded=[SecondDerivativeScenario, HVPScenario], logging=LOGGING
);
