using DifferentiationInterface, DifferentiationInterfaceTest
using FiniteDiff: FiniteDiff

for backend in [AutoFiniteDiff()]
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(
    AutoFiniteDiff(); excluded=[SecondDerivativeScenario, HVPScenario], logging=LOGGING
);
