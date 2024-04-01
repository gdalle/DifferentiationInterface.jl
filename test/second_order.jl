
using Enzyme: Enzyme
using FastDifferentiation: FastDifferentiation
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Tracker: Tracker
using Zygote: Zygote

##

second_order_backends = [
    AutoForwardDiff(),  #
    AutoFastDifferentiation(),
    AutoSparseFastDifferentiation(),
    AutoReverseDiff(),
]

second_order_mixed_backends = [
    # forward over forward
    SecondOrder(AutoForwardDiff(), AutoEnzyme(Enzyme.Forward)),
    # forward over reverse
    SecondOrder(AutoForwardDiff(), AutoZygote()),
    # reverse over forward
    SecondOrder(AutoEnzyme(Enzyme.Reverse), AutoForwardDiff()),
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
    logging=true,
);
