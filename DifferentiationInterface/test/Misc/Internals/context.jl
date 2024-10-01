using DifferentiationInterface
using DifferentiationInterface: Rewrap, with_contexts
using Test

f1(x) = x
g1 = @inferred with_contexts(f1)
@test @inferred g1(4) == 4

f2(x, a, b) = a * x + b
g2 = @inferred with_contexts(f2, Constant(2), Constant(3))
@test @inferred g2(4) == 2 * 4 + 3

contexts = ()
r = @inferred Rewrap()
@test r() == ()

contexts = (Constant(1), Constant(2.0))
r = @inferred Rewrap(contexts...)
@test r(3, 4) == (Constant(3), Constant(4.0))
