using DifferentiationInterface, DifferentiationInterfaceTest
using Enzyme: Enzyme

backends = [
    AutoEnzyme(; mode=nothing),
    AutoEnzyme(; mode=Enzyme.Forward),
    AutoEnzyme(; mode=Enzyme.Reverse),
    AutoSparse(AutoEnzyme(; mode=nothing)),
]

for backend in backends
    @test check_available(backend)
    @test check_twoarg(backend)
    @test !check_hessian(backend; verbose=false)
end

test_differentiation(backends; second_order=false, logging=LOGGING);

test_differentiation(
    AutoEnzyme(; mode=Enzyme.Forward);  # TODO: add more
    correctness=false,
    type_stability=true,
    logging=LOGGING,
);
