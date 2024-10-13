using ADTypes
using DifferentiationInterface
import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using StaticArrays
using Test

@testset "SecondOrder" begin
    backend = SecondOrder(AutoForwardDiff(), AutoZygote())
    @test ADTypes.mode(backend) isa ADTypes.ForwardMode
    @test DI.outer(backend) isa AutoForwardDiff
    @test DI.inner(backend) isa AutoZygote
    @test Bool(DI.inplace_support(backend)) == (
        Bool(DI.inplace_support(DI.inner(backend))) &&
        Bool(DI.inplace_support(DI.outer(backend)))
    )
end

@testset "Sparse" begin
    for backend in [AutoForwardDiff(), AutoZygote()]
        sparse_backend = AutoSparse(backend)
        @test ADTypes.mode(sparse_backend) == ADTypes.mode(backend)
        @test check_available(sparse_backend) == check_available(backend)
        @test DI.inplace_support(sparse_backend) == DI.inplace_support(backend)
    end
end

@testset "Batch size" begin
    @test (@inferred DI.pick_batchsize(AutoZygote(), zeros(2))) == Val(1)
    @test (DI.pick_batchsize(AutoForwardDiff(), zeros(2))) == Val(2)
    @test (DI.pick_batchsize(AutoForwardDiff(), zeros(6))) == Val(6)
    @test (DI.pick_batchsize(AutoForwardDiff(), zeros(100))) == Val(12)
    @test (DI.pick_batchsize(AutoForwardDiff(), @SVector(zeros(2)))) == Val(2)
    @test (DI.pick_batchsize(AutoForwardDiff(), @SVector(zeros(6)))) == Val(6)
    @test (DI.pick_batchsize(AutoForwardDiff(), @SVector(zeros(100)))) == Val(100)
    @test (@inferred DI.pick_batchsize(AutoForwardDiff(; chunksize=4), zeros(2))) == Val(4)
    @test (@inferred DI.pick_batchsize(AutoForwardDiff(; chunksize=4), zeros(6))) == Val(4)
    @test (@inferred DI.pick_batchsize(AutoForwardDiff(; chunksize=4), zeros(100))) ==
        Val(4)
    @test DI.threshold_batchsize(AutoForwardDiff(), 2) isa AutoForwardDiff{nothing}
    @test DI.threshold_batchsize(AutoForwardDiff(; chunksize=4), 2) isa AutoForwardDiff{2}
    @test DI.threshold_batchsize(AutoForwardDiff(; chunksize=4), 6) isa AutoForwardDiff{4}
end
