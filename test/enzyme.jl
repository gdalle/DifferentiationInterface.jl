using DifferentiationInterface
using Enzyme

# see https://github.com/EnzymeAD/Enzyme.jl/issues/1330

@testset "EnzymeForwardBackend" begin
    test_pushforward(
        EnzymeForwardBackend();
        output_type=Union{Number,AbstractVector},  # TODO: remove
        type_stability=true,
    )
    test_jacobian_and_friends(
        EnzymeForwardBackend();
        output_type=Union{Number,AbstractVector},  # TODO: remove
        type_stability=true,
    )
end;

@testset "EnzymeReverseBackend" begin
    test_pullback(EnzymeReverseBackend(); output_type=Number, type_stability=true)
    test_jacobian_and_friends(
        EnzymeReverseBackend(); output_type=Number, type_stability=true
    )
end;
