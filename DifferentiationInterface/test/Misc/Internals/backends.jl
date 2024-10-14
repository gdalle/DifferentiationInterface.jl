using ADTypes
using DifferentiationInterface
using DifferentiationInterface:
    BatchSizeSettings, pick_batchsize, inner, outer, inplace_support, threshold_batchsize
import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using StaticArrays
using Test

BSS = BatchSizeSettings

@testset "SecondOrder" begin
    backend = SecondOrder(AutoForwardDiff(), AutoZygote())
    @test ADTypes.mode(backend) isa ADTypes.ForwardMode
    @test outer(backend) isa AutoForwardDiff
    @test inner(backend) isa AutoZygote
    @test Bool(inplace_support(backend)) ==
        (Bool(inplace_support(inner(backend))) && Bool(inplace_support(outer(backend))))
end

@testset "Sparse" begin
    for backend in [AutoForwardDiff(), AutoZygote()]
        sparse_backend = AutoSparse(backend)
        @test ADTypes.mode(sparse_backend) == ADTypes.mode(backend)
        @test check_available(sparse_backend) == check_available(backend)
        @test inplace_support(sparse_backend) == inplace_support(backend)
    end
end

@testset "Batch size" begin
    @testset "Default" begin
        @test (@inferred pick_batchsize(AutoZygote(), zeros(2))) isa BSS{1,false,true}
        @test (@inferred pick_batchsize(AutoZygote(), zeros(100))) isa BSS{1,false,true}
    end

    @testset "ForwardDiff" begin
        @test (pick_batchsize(AutoForwardDiff(), zeros(2))) isa BSS{2,true,true}
        @test (pick_batchsize(AutoForwardDiff(), zeros(6))) isa BSS{6,true,true}
        @test (pick_batchsize(AutoForwardDiff(), zeros(12))) isa BSS{12,true,true}
        @test (pick_batchsize(AutoForwardDiff(), zeros(24))) isa BSS{12,false,true}
        @test (pick_batchsize(AutoForwardDiff(), zeros(100))) isa BSS{12,false,false}
        @test (@inferred pick_batchsize(AutoForwardDiff(), @SVector(zeros(2)))) isa
            BSS{2,true,true}
        @test (@inferred pick_batchsize(AutoForwardDiff(), @SVector(zeros(6)))) isa
            BSS{6,true,true}
        @test (@inferred pick_batchsize(AutoForwardDiff(), @SVector(zeros(100)))) isa
            BSS{100,true,true}
        @test_throws ArgumentError pick_batchsize(AutoForwardDiff(; chunksize=4), zeros(2))
        @test (@inferred pick_batchsize(AutoForwardDiff(; chunksize=4), zeros(6))) isa
            BSS{4}
        @test (@inferred pick_batchsize(AutoForwardDiff(; chunksize=4), zeros(100))) isa
            BSS{4}
        @test threshold_batchsize(AutoForwardDiff(), 2) isa AutoForwardDiff{nothing}
        @test threshold_batchsize(AutoForwardDiff(; chunksize=4), 2) isa AutoForwardDiff{2}
        @test threshold_batchsize(AutoForwardDiff(; chunksize=4), 6) isa AutoForwardDiff{4}
    end
end
