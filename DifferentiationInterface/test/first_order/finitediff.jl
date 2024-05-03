using DifferentiationInterface, DifferentiationInterfaceTest
using FiniteDiff: FiniteDiff

backends = [AutoFiniteDiff()]

for backend in backends
    @test check_available(backend)
    @test check_twoarg(backend)
    @test check_hessian(backend)
end

test_differentiation(
    backends; excluded=[SecondDerivativeScenario, HVPScenario], logging=LOGGING
);
