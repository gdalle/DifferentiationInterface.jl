using DifferentiationInterface.DifferentiationTest: backend_string

using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using Enzyme: Enzyme
using Zygote: Zygote

SECOND_ORDER_BACKENDS = Dict(
    "forward/forward" => [
        SecondOrder(AutoEnzyme(Enzyme.Forward), AutoForwardDiff()),
        SecondOrder(AutoForwardDiff(), AutoEnzyme(Enzyme.Forward)),
    ],
    "forward/reverse" => [SecondOrder(AutoForwardDiff(), AutoZygote())],
    "reverse/forward" => [],
)

@testset verbose = true "Second order" begin
    @testset verbose = true "$second_order_mode" for (second_order_mode, backends) in
                                                     pairs(SECOND_ORDER_BACKENDS)
        @info "Testing $second_order_mode..."
        @time @testset "$(backend_string(backend))" for backend in backends
            test_differentiation(backend; first_order=false, type_stability=false)
        end
    end
end;
