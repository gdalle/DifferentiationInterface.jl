using DifferentiationInterface
using Enzyme

@testset "EnzymeForwardBackend" begin
    test_pushforward(EnzymeForwardBackend(); type_stability=true)
    test_jacobian(EnzymeForwardBackend(); type_stability=true)
end
@testset "EnzymeReverseBackend" begin
    test_pullback(EnzymeReverseBackend(); output_type=Number, type_stability=true)
    test_jacobian(EnzymeReverseBackend(); output_type=Number, type_stability=true)
end
