##

using Enzyme: Enzyme
using FastDifferentiation: FastDifferentiation
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using SparseDiffTools: SparseDiffTools, AutoSparseEnzyme
using Symbolics: Symbolics
using Zygote: Zygote

##

sparse_backends = [
    AutoSparseFastDifferentiation(),
    AutoSparseForwardDiff(),
    AutoSparseFiniteDiff(),
    AutoSparseZygote(),
    AutoSparseEnzyme(),
]

##

abs2diff(x) = abs2.(diff(x))
abs2diff!(y, x) = y .= abs2.(diff(x))

function abs2diff_jacobian(x)
    return SparseArrays.spdiagm(
        length(x) - 1, length(x), 0 => -2 * diff(x), 1 => 2 * diff(x)
    )
end

test_differentiation(
    sparse_backends,
    [
        JacobianScenario(abs2diff; x=rand(5), ref=abs2diff_jacobian),
        JacobianScenario(
            abs2diff!; x=rand(5), y=zeros(4), ref=jacobian = abs2diff_jacobian
        ),
    ];
    sparsity=true,
    logging=get(ENV, "CI", "false") == "false",
)
