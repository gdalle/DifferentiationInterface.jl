using Enzyme: Enzyme
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff

second_order_backends = [AutoForwardDiff(), AutoReverseDiff()]

second_order_mixed_backends = [
    SecondOrder(AutoEnzyme(Enzyme.Forward), AutoForwardDiff()),
    SecondOrder(AutoForwardDiff(), AutoEnzyme(Enzyme.Forward)),
    SecondOrder(AutoForwardDiff(), AutoZygote()),
]

for backend in vcat(second_order_backends, second_order_mixed_backends)
    @test check_hessian(backend)
end

test_differentiation(
    vcat(second_order_backends, second_order_mixed_backends);
    first_order=false,
    second_order=true,
);
