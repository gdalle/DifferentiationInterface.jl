using DifferentiationInterfaceTest: AutoZeroForward, AutoZeroReverse

@test check_available(AutoZeroForward())
@test check_available(AutoZeroReverse())

## Correctness (vs oneself)

for backend in [AutoZeroForward(), AutoZeroReverse()]
    test_differentiation(
        backend, all_operators(), default_scenarios(); correctness=backend, logging=true
    )
end

for backend in [
    SecondOrder(AutoZeroForward(), AutoZeroReverse()),
    SecondOrder(AutoZeroReverse(), AutoZeroForward()),
]
    test_differentiation(
        backend,
        all_operators(),
        default_scenarios();
        correctness=backend,
        first_order=false,
        logging=true,
    )
end

## Type stability

test_differentiation(
    AutoZeroForward();
    correctness=false,
    type_stability=true,
    excluded=[pullback],
    logging=true,
)
test_differentiation(
    AutoZeroReverse();
    correctness=false,
    type_stability=true,
    excluded=[pushforward],
    logging=true,
)
test_differentiation(
    [
        SecondOrder(AutoZeroForward(), AutoZeroReverse()),
        SecondOrder(AutoZeroReverse(), AutoZeroForward()),
    ];
    correctness=false,
    type_stability=true,
    first_order=false,
    logging=true,
)

## Call count

test_differentiation(
    AutoZeroForward();
    correctness=false,
    call_count=true,
    excluded=[gradient, pullback],
    logging=true,
);

test_differentiation(
    AutoZeroReverse();
    correctness=false,
    call_count=true,
    excluded=[derivative, pushforward],
    logging=true,
);

## Benchmark

data = benchmark_differentiation([AutoZeroForward(), AutoZeroReverse()]; logging=true);

df = DataFrames.DataFrame(pairs(data)...)

## Weird arrays

for backend in [AutoZeroForward(), AutoZeroReverse()]
    test_differentiation(
        backend, all_operators(), gpu_scenarios(); correctness=backend, logging=true
    )
    # copyto!(col, col) fails on static arrays
    test_differentiation(
        backend,
        all_operators(),
        static_scenarios();
        correctness=backend,
        excluded=[jacobian, hessian],
        logging=true,
    )
    # stack fails on component vectors
    test_differentiation(
        backend,
        all_operators(),
        component_scenarios();
        correctness=backend,
        excluded=[hessian],
        logging=true,
    )
end
