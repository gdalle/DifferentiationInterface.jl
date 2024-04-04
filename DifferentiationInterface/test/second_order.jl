second_order_backends = [
    AutoForwardDiff(),  #
    AutoPolyesterForwardDiff(; chunksize=1),  #
    AutoFastDifferentiation(),
    AutoReverseDiff(),
]

second_order_mixed_backends = [
    # forward over forward
    SecondOrder(AutoForwardDiff(), AutoEnzyme(Enzyme.Forward)),
    # forward over reverse
    SecondOrder(AutoForwardDiff(), AutoZygote()),
    # reverse over forward
    SecondOrder(AutoZygote(), AutoFiniteDiff()),
    # reverse over reverse
    SecondOrder(AutoReverseDiff(), AutoZygote()),
]

##

for backend in vcat(second_order_backends, second_order_mixed_backends)
    check_hessian(backend)
end

test_differentiation(
    vcat(second_order_backends, second_order_mixed_backends);
    first_order=false,
    second_order=true,
    logging=get(ENV, "CI", "false") == "false",
);

## only Hessian

only_hessian_backends = [AutoFiniteDiff(), AutoZygote()]

test_differentiation(
    only_hessian_backends;
    first_order=false,
    second_order=true,
    excluded=[SecondDerivativeScenario, HVPScenario],
    logging=get(ENV, "CI", "false") == "false",
);
