using DifferentiationInterface
using ForwardDiff
using Test

f1(x::Real) = exp(x)
f2(x::Real) = [exp(x), exp(2x)]
f3(x::AbstractArray) = sum(exp, x)
f4(x::AbstractArray) = exp.(x)

x1 = x2 = 2.0
x3 = x4 = [1.0, 2.0]

dx1 = dx2 = 1.0
dx3 = dx4 = [0.0, 1.0]

dy1 = dy3 = 0.0
dy2 = dy4 = [0.0, 0.0]

dy1_true = exp(2.0)
dy2_true = [exp(2.0), 2exp(4.0)]
dy3_true = exp(2.0)
dy4_true = [0.0, exp(2.0)]

backend = ForwardDiffBackend()

@test only(jvp!((dy1,), backend, f1, (x1,), (dx1,))) ≈ dy1_true
@test only(jvp!((dy2,), backend, f2, (x2,), (dx2,))) ≈ dy2_true
@test only(jvp!((dy3,), backend, f3, (x3,), (dx3,))) ≈ dy3_true
@test only(jvp!((dy4,), backend, f4, (x4,), (dx4,))) ≈ dy4_true
