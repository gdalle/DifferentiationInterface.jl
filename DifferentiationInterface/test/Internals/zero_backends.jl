using DifferentiationInterface
using DifferentiationInterface: AutoZeroForward, AutoZeroReverse
using DifferentiationInterfaceTest
using ComponentArrays: ComponentArrays
using JLArrays: JLArrays
using StaticArrays: StaticArrays
using Test

LOGGING = get(ENV, "CI", "false") == "false"

zero_backends = [AutoZeroForward(), AutoZeroReverse()]

for backend in zero_backends
    @test check_available(backend)
    @test check_inplace(backend)
end

## Type stability

test_differentiation(
    zero_backends,
    zero.(default_scenarios());
    correctness=false,
    type_stability=true,
    # excluded=[:second_derivative],
    logging=LOGGING,
)

test_differentiation(
    [
        SecondOrder(AutoZeroForward(), AutoZeroReverse()),
        SecondOrder(AutoZeroReverse(), AutoZeroForward()),
    ],
    default_scenarios();
    correctness=false,
    type_stability=true,
    first_order=false,
    logging=LOGGING,
)

## Weird arrays

test_differentiation(
    [AutoZeroForward(), AutoZeroReverse()],
    zero.(vcat(component_scenarios(), static_scenarios()));
    correctness=true,
    logging=LOGGING,
)

if VERSION >= v"1.10"
    test_differentiation(
        [AutoZeroForward(), AutoZeroReverse()],
        zero.(gpu_scenarios());
        correctness=true,
        logging=LOGGING,
    )
end
