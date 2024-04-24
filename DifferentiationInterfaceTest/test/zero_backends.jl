@test check_available(AutoZeroForward())
@test check_available(AutoZeroReverse())
@test check_twoarg(AutoZeroForward())
@test check_twoarg(AutoZeroReverse())

## Correctness (vs oneself)

for backend in [AutoZeroForward(), AutoZeroReverse()]
    test_differentiation(
        backend,
        default_scenarios();
        correctness=true,
        ref_backend=backend,
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
        correctness=true,
        first_order=false,
        ref_backend=backend,
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

## Benchmark

data1 = benchmark_differentiation(
    [AutoZeroForward(), AutoZeroReverse()]; logging=get(ENV, "CI", "false") == "false"
);

data2 = benchmark_differentiation(
    [SecondOrder(AutoZeroForward(), AutoZeroReverse())];
    first_order=false,
    logging=get(ENV, "CI", "false") == "false",
);

df1 = DataFrames.DataFrame(data1)
df2 = DataFrames.DataFrame(data2)

## Weird arrays

for backend in [AutoZeroForward(), AutoZeroReverse()]
    test_differentiation(
        backend,
        gpu_scenarios();
        correctness=true,
        ref_backend=backend,
        logging=get(ENV, "CI", "false") == "false",
    )
    test_differentiation(
        backend,
        static_scenarios();
        correctness=true,
        ref_backend=backend,
        logging=get(ENV, "CI", "false") == "false",
    )
    # stack fails on component vectors
    test_differentiation(
        backend,
        component_scenarios();
        correctness=true,
        excluded=[HessianScenario],
        ref_backend=backend,
        logging=get(ENV, "CI", "false") == "false",
    )
end
