using ADTypes
using ADTypes: mode
using DifferentiationInterface
using DifferentiationInterface:
    inner, outer, inplace_support, pushforward_performance, pullback_performance, hvp_mode
import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using Test

@testset "SecondOrder" begin
    backend = SecondOrder(AutoForwardDiff(), AutoZygote())
    @test ADTypes.mode(backend) isa ADTypes.ForwardMode
    @test outer(backend) isa AutoForwardDiff
    @test inner(backend) isa AutoZygote
    @test mode(backend) isa ADTypes.ForwardMode
    @test Bool(inplace_support(backend)) ==
        (Bool(inplace_support(inner(backend))) && Bool(inplace_support(outer(backend))))
    @test_throws ArgumentError pushforward_performance(backend)
    @test_throws ArgumentError pullback_performance(backend)
end

@testset "Sparse" begin
    for dense_backend in [AutoForwardDiff(), AutoZygote()]
        backend = AutoSparse(dense_backend)
        @test ADTypes.mode(backend) == ADTypes.mode(dense_backend)
        @test check_available(backend) == check_available(dense_backend)
        @test inplace_support(backend) == inplace_support(dense_backend)
        @test_throws ArgumentError pushforward_performance(backend)
        @test_throws ArgumentError pullback_performance(backend)
        @test_throws ArgumentError hvp_mode(backend)
    end
end
