using DifferentiationInterface
using Test

@test_throws ArgumentError Tangents()

t = Tangents([2.0])
@test NTuple(t) == ([2.0],)
@test length(t) == 1
@test eltype(t) == Vector{Float64}
@test only(t) == [2.0]
@test copyto!(map(zero, t), t) ≈ t

t = Tangents(2.0, 4.0, 6.0)
@test NTuple(t) == (2.0, 4.0, 6.0)
@test length(t) == 3
@test eltype(t) == Float64
@test t[begin] == first(t) == 2.0
@test t[end] == last(t) == 6.0
@test collect(t) == [2.0, 4.0, 6.0]
@test map(abs2, t) == Tangents(4.0, 16.0, 36.0)
