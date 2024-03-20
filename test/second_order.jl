using ADTypes
using DifferentiationInterface
using DifferentiationInterface.DifferentiationTest

using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using Enzyme: Enzyme
using ReverseDiff: ReverseDiff
using Zygote: Zygote

using JET: JET
using Test

@testset verbose = true "Forward over forward" begin
    @testset "$(backend_string(backend))" for backend in [
        SecondOrder(AutoEnzyme(Enzyme.Forward), AutoFiniteDiff()),
        SecondOrder(AutoEnzyme(Enzyme.Forward), AutoForwardDiff()),
        SecondOrder(AutoForwardDiff(), AutoEnzyme(Enzyme.Forward)),
        SecondOrder(AutoForwardDiff(), AutoFiniteDiff()),
        SecondOrder(AutoFiniteDiff(), AutoEnzyme(Enzyme.Forward)),
        SecondOrder(AutoFiniteDiff(), AutoForwardDiff()),
    ]
        test_operators(backend; first_order=false, type_stability=false)
    end
end;

@testset verbose = true "Forward over reverse" begin
    @testset "$(backend_string(backend))" for backend in [
        # SecondOrder(AutoEnzyme(Enzyme.Reverse), AutoEnzyme(Enzyme.Forward)),
        SecondOrder(AutoEnzyme(Enzyme.Reverse), AutoFiniteDiff()),
        # SecondOrder(AutoEnzyme(Enzyme.Reverse), AutoForwardDiff()),
        SecondOrder(AutoReverseDiff(), AutoEnzyme(Enzyme.Forward)),
        SecondOrder(AutoReverseDiff(), AutoFiniteDiff()),
        SecondOrder(AutoReverseDiff(), AutoForwardDiff()),
        SecondOrder(AutoZygote(), AutoEnzyme(Enzyme.Forward)),
        SecondOrder(AutoZygote(), AutoFiniteDiff()),
        SecondOrder(AutoZygote(), AutoForwardDiff()),
    ]
        test_operators(backend; first_order=false, type_stability=false)
    end
end;

@testset verbose = true "Reverse over forward" begin
    @testset "$(backend_string(backend))" for backend in [
        # SecondOrder(AutoEnzyme(Enzyme.Forward), AutoEnzyme(Enzyme.Reverse)),
        # SecondOrder(AutoEnzyme(Enzyme.Forward), AutoReverseDiff()),
        # SecondOrder(AutoEnzyme(Enzyme.Forward), AutoZygote()),
        # SecondOrder(AutoFiniteDiff(), AutoEnzyme(Enzyme.Reverse)),
        SecondOrder(AutoFiniteDiff(), AutoReverseDiff()),
        SecondOrder(AutoFiniteDiff(), AutoZygote()),
        SecondOrder(AutoForwardDiff(), AutoEnzyme(Enzyme.Reverse)),
        # SecondOrder(AutoForwardDiff(), AutoReverseDiff()),
        # SecondOrder(AutoForwardDiff(), AutoZygote()),
    ]
        test_operators(backend; first_order=false, type_stability=false)
    end
end;

@testset verbose = true "Reverse over reverse" begin
    @testset "$(backend_string(backend))" for backend in [
        # SecondOrder(AutoEnzyme(Enzyme.Reverse), AutoEnzyme(Enzyme.Reverse)),
        # SecondOrder(AutoEnzyme(Enzyme.Reverse), AutoReverseDiff()),
        # SecondOrder(AutoEnzyme(Enzyme.Reverse), AutoZygote()),
        # SecondOrder(AutoReverseDiff(), AutoEnzyme(Enzyme.Reverse)),
        SecondOrder(AutoReverseDiff(), AutoReverseDiff()),
        # SecondOrder(AutoReverseDiff(), AutoZygote()),
        SecondOrder(AutoZygote(), AutoEnzyme(Enzyme.Reverse)),
        SecondOrder(AutoZygote(), AutoReverseDiff()),
        SecondOrder(AutoZygote(), AutoZygote()),
    ]
        test_operators(backend; first_order=false, type_stability=false)
    end
end;
