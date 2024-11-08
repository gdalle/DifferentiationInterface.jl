using ADTypes
using DifferentiationInterface
using DifferentiationInterface:
    BatchSizeSettings, pick_batchsize, reasonable_batchsize, threshold_batchsize
import DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using StaticArrays
using Test

BSS = BatchSizeSettings

@testset "Default" begin
    @test (@inferred pick_batchsize(AutoZygote(), zeros(2))) isa BSS{1,false,true}
    @test (@inferred pick_batchsize(AutoZygote(), zeros(100))) isa BSS{1,false,true}
    @test_throws ArgumentError pick_batchsize(AutoSparse(AutoZygote()), zeros(2))
    @test_throws ArgumentError pick_batchsize(
        SecondOrder(AutoZygote(), AutoZygote()), zeros(2)
    )
    @test_throws ArgumentError pick_batchsize(
        MixedMode(AutoForwardDiff(), AutoZygote()), zeros(2)
    )
end

@testset "ForwardDiff (adaptive)" begin
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
end

@testset "ForwardDiff (fixed)" begin
    @test_throws ArgumentError pick_batchsize(AutoForwardDiff(; chunksize=4), zeros(2))
    @test_throws ArgumentError pick_batchsize(
        AutoForwardDiff(; chunksize=4), @SVector(zeros(2))
    )
    @test pick_batchsize(AutoForwardDiff(; chunksize=4), zeros(6)) isa BSS{4}
    @test pick_batchsize(AutoForwardDiff(; chunksize=4), zeros(100)) isa BSS{4}
    BSS{4,true,true}
    @test pick_batchsize(AutoForwardDiff(; chunksize=4), zeros(99)) isa BSS{4}
    BSS{4,true,false}
    @test (@inferred pick_batchsize(AutoForwardDiff(; chunksize=4), @SVector(zeros(6)))) isa
        BSS{4,false,false}
    @test (@inferred pick_batchsize(
        AutoForwardDiff(; chunksize=4), @SVector(zeros(100))
    )) isa BSS{4,false,true}
end

@testset "Thresholding" begin
    @test threshold_batchsize(AutoForwardDiff(), 2) isa AutoForwardDiff{nothing}
    @test threshold_batchsize(AutoForwardDiff(; chunksize=4), 2) isa AutoForwardDiff{2}
    @test threshold_batchsize(AutoForwardDiff(; chunksize=4), 6) isa AutoForwardDiff{4}
    @test threshold_batchsize(AutoSparse(AutoForwardDiff(; chunksize=4)), 2) isa
        AutoSparse{<:AutoForwardDiff{2}}
    @test threshold_batchsize(
        SecondOrder(AutoForwardDiff(; chunksize=4), AutoForwardDiff(; chunksize=3)), 6
    ) isa SecondOrder{<:AutoForwardDiff{4},<:AutoForwardDiff{3}}
    @test threshold_batchsize(
        SecondOrder(AutoForwardDiff(; chunksize=4), AutoForwardDiff(; chunksize=3)), 2
    ) isa SecondOrder{<:AutoForwardDiff{2},<:AutoForwardDiff{2}}
    @test threshold_batchsize(
        SecondOrder(AutoForwardDiff(; chunksize=1), AutoForwardDiff(; chunksize=3)), 2
    ) isa SecondOrder{<:AutoForwardDiff{1},<:AutoForwardDiff{2}}
    @test threshold_batchsize(
        SecondOrder(AutoForwardDiff(; chunksize=4), AutoForwardDiff(; chunksize=1)), 2
    ) isa SecondOrder{<:AutoForwardDiff{2},<:AutoForwardDiff{1}}
end

@testset "Reasonable" begin
    for Bmax in 1:5
        @test all(<=(Bmax), reasonable_batchsize.(1:10, Bmax))
        @test issorted(div.(1:10, reasonable_batchsize.(1:10, Bmax), RoundUp))
        if Bmax > 2
            @test reasonable_batchsize(Bmax + 1, Bmax) < Bmax
        end
    end
end
