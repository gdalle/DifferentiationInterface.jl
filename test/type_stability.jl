using Enzyme: Enzyme
using ForwardDiff: ForwardDiff

type_stable_backends = [AutoForwardDiff(), AutoEnzyme(Enzyme.Reverse)]

test_differentiation(
    type_stable_backends;
    correctness=false,
    type_stability=true,
    second_order=false,
    logging=get(ENV, "CI", "false") == "false",
);
