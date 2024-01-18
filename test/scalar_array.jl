using DifferentiationInterface
using Enzyme
using ForwardDiff
using Test

# all vectors have length 2

f1(x::Real) = exp(2x)
f2(x::Real) = [exp(2x), exp(3x)]
f3(x::AbstractVector) = exp(2x[1]) + exp(3x[2])
f4(x::AbstractVector) = [exp(2x[1]), exp(3x[2])]

x1 = x2 = 1.0
x3 = x4 = [1.0, 2.0]

## JVP

dx1 = dx2 = 5.0
dx3 = dx4 = [0.0, 5.0]

dy1_true = 2exp(2x1) * dx1
dy2_true = [2exp(2x1), 3exp(3x2)] .* dx2
dy3_true = 3exp(3x3[2]) * dx3[2]
dy4_true = [0.0, 3exp(3x4[2])] .* dx4[2]

## VJP

dy1 = dy3 = 5.0
dy2 = dy4 = [0.0, 5.0]

dx1_true = 2exp(2x1) * dy1
dx2_true = 3exp(3x2) * dy2[2]
dx3_true = [2exp(2x3[1]), 3exp(3x3[2])] .* dy3
dx4_true = [0.0, 3exp(3x4[2])] .* dy4[2]

## ForwardDiff

backend = ForwardDiffBackend()

@testset "ForwardDiff - JVP" begin
    @test jvp!(0.0, backend, f1, x1, dx1) ≈ dy1_true
    @test jvp!(zeros(2), backend, f2, x2, dx2) ≈ dy2_true
    @test jvp!(0.0, backend, f3, x3, dx3) ≈ dy3_true
    @test jvp!(zeros(2), backend, f4, x4, dx4) ≈ dy4_true
end

@testset "ForwardDiff - VJP" begin
    @test vjp!(0.0, backend, f1, x1, dy1) ≈ dx1_true
    @test vjp!(0.0, backend, f2, x2, dy2) ≈ dx2_true
    @test vjp!(zeros(2), backend, f3, x3, dy3) ≈ dx3_true
    @test vjp!(zeros(2), backend, f4, x4, dy4) ≈ dx4_true
end

## Enzyme

backend = EnzymeBackend()

@testset "Enzyme - JVP" begin
    @test jvp!(0.0, backend, f1, x1, dx1) ≈ dy1_true
    @test jvp!(zeros(2), backend, f2, x2, dx2) ≈ dy2_true
    @test jvp!(0.0, backend, f3, x3, dx3) ≈ dy3_true
    @test jvp!(zeros(2), backend, f4, x4, dx4) ≈ dy4_true
end

@testset "Enzyme - VJP" begin
    # only scalar returns
    @test vjp!(0.0, backend, f1, x1, dy1) ≈ dx1_true
    @test vjp!(zeros(2), backend, f3, x3, dy3) ≈ dx3_true
end
