using ADTypes
using DifferentiationInterface
import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using Test

@testset "SecondOrder" begin
    backend = SecondOrder(AutoForwardDiff(), AutoZygote())
    @test ADTypes.mode(backend) isa ADTypes.ForwardMode
    @test DifferentiationInterface.outer(backend) isa AutoForwardDiff
    @test DifferentiationInterface.inner(backend) isa AutoZygote
end

@testset "Sparse" begin
    for backend in [AutoForwardDiff(), AutoZygote()]
        sparse_backend = AutoSparse(backend)
        @test ADTypes.mode(sparse_backend) == ADTypes.mode(backend)
        @test check_available(sparse_backend) == check_available(backend)
        @test DI.inplace_support(sparse_backend) == DI.inplace_support(backend)
        @test DI.pushforward_performance(sparse_backend) ==
            DI.pushforward_performance(backend)
        @test DI.pullback_performance(sparse_backend) == DI.pullback_performance(backend)
    end

    for backend in [
        SecondOrder(AutoForwardDiff(), AutoZygote()),
        SecondOrder(AutoZygote(), AutoForwardDiff()),
    ]
        sparse_backend = AutoSparse(backend)
        @test ADTypes.mode(sparse_backend) == ADTypes.mode(backend)
        @test DI.hvp_mode(sparse_backend) == DI.hvp_mode(backend)
    end
end

@testset "Batch size" begin
    @test DI.pick_batchsize(AutoZygote(), 2) == Val(1)
    @test DI.pick_batchsize(AutoForwardDiff(), 2) == Val(2)
    @test DI.pick_batchsize(AutoForwardDiff(), 6) == Val(6)
    @test DI.pick_batchsize(AutoForwardDiff(), 100) == Val(12)
    @test DI.pick_batchsize(AutoForwardDiff(; chunksize=4), 2) == Val(4)
    @test DI.pick_batchsize(AutoForwardDiff(; chunksize=4), 6) == Val(4)
    @test DI.pick_batchsize(AutoForwardDiff(; chunksize=4), 100) == Val(4)
    @test DI.threshold_batchsize(AutoForwardDiff(), 2) isa AutoForwardDiff{nothing}
    @test DI.threshold_batchsize(AutoForwardDiff(; chunksize=4), 2) isa AutoForwardDiff{2}
    @test DI.threshold_batchsize(AutoForwardDiff(; chunksize=4), 6) isa AutoForwardDiff{4}
    @test DI.pick_batchsize(AutoSparse(AutoForwardDiff(; chunksize=4)), 10) == Val(4)
end
