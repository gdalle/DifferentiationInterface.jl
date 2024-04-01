using DifferentiationInterfaceTest: AutoZeroForward, AutoZeroReverse

@test check_available(AutoZeroForward())
@test check_available(AutoZeroReverse())

## Correctness (vs oneself)

for backend in [AutoZeroForward(), AutoZeroReverse()]
    test_differentiation(
        backend,
        default_scenarios();
        correctness=backend,
        logging=get(ENV, "CI", "false") == "false",
    )
end

for backend in [
    SecondOrder(AutoZeroForward(), AutoZeroReverse()),
    SecondOrder(AutoZeroReverse(), AutoZeroForward()),
]
    test_differentiation(
        backend,
        default_scenarios();
        correctness=backend,
        first_order=false,
        logging=get(ENV, "CI", "false") == "false",
    )
end

## Type stability

test_differentiation(
    [AutoZeroForward(), AutoZeroReverse()];
    correctness=false,
    type_stability=true,
    logging=get(ENV, "CI", "false") == "false",
)

test_differentiation(
    [
        SecondOrder(AutoZeroForward(), AutoZeroReverse()),
        SecondOrder(AutoZeroReverse(), AutoZeroForward()),
    ];
    correctness=false,
    type_stability=true,
    first_order=false,
    logging=get(ENV, "CI", "false") == "false",
)

## Call count

test_differentiation(
    AutoZeroForward();
    correctness=false,
    call_count=true,
    logging=get(ENV, "CI", "false") == "false",
    excluded=[GradientScenario],
);

test_differentiation(
    AutoZeroReverse();
    correctness=false,
    call_count=true,
    logging=get(ENV, "CI", "false") == "false",
    excluded=[DerivativeScenario],
);

## Benchmark

data = benchmark_differentiation(
    [AutoZeroForward(), AutoZeroReverse()]; logging=get(ENV, "CI", "false") == "false"
);

df = DataFrames.DataFrame(data)

## Weird arrays

for backend in [AutoZeroForward(), AutoZeroReverse()]
    test_differentiation(
        backend,
        gpu_scenarios();
        correctness=backend,
        logging=get(ENV, "CI", "false") == "false",
    )
    # copyto!(col, col) fails on static arrays
    test_differentiation(
        backend,
        static_scenarios();
        correctness=backend,
        excluded=[JacobianScenario, HessianScenario],
        logging=get(ENV, "CI", "false") == "false",
    )
    # stack fails on component vectors
    test_differentiation(
        backend,
        component_scenarios();
        correctness=backend,
        excluded=[HessianScenario],
        logging=get(ENV, "CI", "false") == "false",
    )
end
