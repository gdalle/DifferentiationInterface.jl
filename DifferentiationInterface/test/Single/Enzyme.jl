using DifferentiationInterface, DifferentiationInterfaceTest
using Enzyme: Enzyme

for backend in [
    AutoEnzyme(; mode=nothing),
    AutoEnzyme(; mode=Enzyme.Forward),
    AutoEnzyme(; mode=Enzyme.Reverse),
    AutoSparse(AutoEnzyme(; mode=nothing)),
]
    @test check_available(backend)
    @test check_twoarg(backend)
    @test !check_hessian(backend; verbose=false)
end

test_differentiation(backends; second_order=false, logging=LOGGING);

test_differentiation(
    MyAutoSparse(AutoEnzyme(Enzyme.Reverse)),
    sparse_scenarios();
    second_order=false,
    sparsity=true,
    logging=LOGGING,
);

test_differentiation(
    AutoEnzyme(; mode=Enzyme.Forward);  # TODO: add more
    correctness=false,
    type_stability=true,
    second_order=false,
    logging=LOGGING,
);
