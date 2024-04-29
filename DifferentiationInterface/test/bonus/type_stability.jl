type_stable_backends = [
    AutoForwardDiff(),  #
    AutoEnzyme(Enzyme.Forward),
    # AutoEnzyme(Enzyme.Reverse),  # TODO: add it back
]

test_differentiation(
    type_stable_backends;
    correctness=false,
    type_stability=true,
    second_order=false,
    logging=get(ENV, "CI", "false") == "false",
);
