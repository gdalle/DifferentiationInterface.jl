using ADTypes: jacobian_sparsity, hessian_sparsity
using DifferentiationInterface
using ForwardDiff: ForwardDiff
using Enzyme: Enzyme
using LinearAlgebra
using SparseArrays
using StableRNGs
using Test

rng = StableRNG(63)

const Jc = sprand(rng, Bool, 10, 20, 0.3)
const Hc = sparse(Symmetric(sprand(rng, Bool, 20, 20, 0.3)))

f(x) = Jc * x

function f!(y, x)
    mul!(y, Jc, x)
    return nothing
end

g(x) = dot(x, Hc, x)

@testset verbose = true "$(typeof(backend))" for backend in [
    AutoEnzyme(; mode=Enzyme.Reverse), AutoForwardDiff()
]
    @testset "$method" for method in (:iterative, :direct)
        detector = DenseSparsityDetector(backend; atol=1e-5, method)
        @test J == jacobian_sparsity(f, rand(size(J, 2)), detector)
        @test J == jacobian_sparsity(f!, zeros(size(J, 1)), rand(size(J, 2)), detector)
        if backend isa AutoForwardDiff
            @test H == hessian_sparsity(g, rand(size(H, 2)), detector)
        end
    end
end
