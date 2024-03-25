include("test_imports.jl")

using DifferentiationInterface.DifferentiationTest: AutoZeroForward, AutoZeroReverse

@test check_available(AutoZeroForward())
@test check_available(AutoZeroReverse())

## Error-free & type-stability

test_differentiation(
    [AutoZeroForward(), AutoZeroReverse()];
    correctness=false,
    error_free=true,
    type_stability=true,
);

test_differentiation(
    [AutoZeroForward(), AutoZeroReverse()],
    all_operators(),
    weird_array_scenarios(; static=true, component=false, gpu=true);
    correctness=false,
    error_free=true,
);

test_differentiation(
    [AutoZeroForward(), AutoZeroReverse()],
    all_operators(),
    weird_array_scenarios(; static=false, component=true, gpu=false);
    correctness=false,
    error_free=true,
    excluded=[hessian],
);

## Call count

test_differentiation(
    AutoZeroForward(); correctness=false, call_count=true, excluded=[gradient]
);

test_differentiation(
    AutoZeroReverse(); correctness=false, call_count=true, excluded=[derivative]
);

## Allocations

test_differentiation(
    [AutoZeroForward(), AutoZeroReverse()];
    correctness=false,
    allocations=true,
    excluded=[jacobian],
);

data = test_differentiation(
    [AutoZeroForward(), AutoZeroReverse()]; correctness=false, benchmark=true
);

df = DataFrames.DataFrame(pairs(data)...)
