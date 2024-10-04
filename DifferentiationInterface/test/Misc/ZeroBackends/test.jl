using DifferentiationInterface
using DifferentiationInterface: AutoZeroForward, AutoZeroReverse
using DifferentiationInterfaceTest
using ComponentArrays: ComponentArrays
using JLArrays: JLArrays
using SparseMatrixColorings
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
    AutoZeroForward(),
    default_scenarios(; include_batchified=false, include_constantified=true);
    correctness=false,
    type_stability=(; preparation=true, prepared_op=true, unprepared_op=true),
    logging=LOGGING,
)

test_differentiation(
    AutoZeroReverse(),
    default_scenarios(; include_batchified=false, include_constantified=true);
    correctness=false,
    # TODO: set unprepared_op=true after ignoring DataFrames
    type_stability=(; preparation=true, prepared_op=true, unprepared_op=false),
    logging=LOGGING,
)

test_differentiation(
    [
        SecondOrder(AutoZeroForward(), AutoZeroReverse()),
        SecondOrder(AutoZeroReverse(), AutoZeroForward()),
    ],
    default_scenarios(; include_batchified=false, include_constantified=true);
    correctness=false,
    type_stability=(; preparation=true, prepared_op=true, unprepared_op=true),
    first_order=false,
    logging=LOGGING,
)

test_differentiation(
    AutoSparse.(zero_backends, coloring_algorithm=GreedyColoringAlgorithm()),
    default_scenarios(; include_constantified=true);
    correctness=false,
    type_stability=(; preparation=true, prepared_op=true, unprepared_op=false),
    excluded=[:pushforward, :pullback, :gradient, :derivative, :hvp, :second_derivative],
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
