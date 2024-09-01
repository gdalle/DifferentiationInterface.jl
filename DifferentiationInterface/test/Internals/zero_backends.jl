using DifferentiationInterface
using DifferentiationInterface: AutoZeroForward, AutoZeroReverse
using DifferentiationInterfaceTest
using Test

LOGGING = get(ENV, "CI", "false") == "false"

zero_backends = [AutoZeroForward(), AutoZeroReverse()]

for backend in zero_backends
    @test check_available(backend)
    @test check_twoarg(backend)
end

## Type stability

test_differentiation(
    zero_backends,
    default_scenarios();
    correctness=false,
    type_stability=true,
    excluded=[:second_derivative],
    logging=LOGGING,
)

test_differentiation(
    [
        SecondOrder(AutoZeroForward(), AutoZeroReverse()),
        SecondOrder(AutoZeroReverse(), AutoZeroForward()),
    ],
    default_scenarios(; linalg=false);
    correctness=false,
    type_stability=true,
    first_order=false,
    logging=LOGGING,
)
