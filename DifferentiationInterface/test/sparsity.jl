sparse_backends = [
    AutoSparseFastDifferentiation(),
    AutoSparseForwardDiff(),
    AutoSparseFiniteDiff(),
    AutoSparseZygote(),
]

sparse_second_order_backends = [
    AutoSparseFastDifferentiation(),
    SecondOrder(AutoSparseForwardDiff(), AutoZygote()),
    SecondOrder(AutoSparseFiniteDiff(), AutoZygote()),
]

##

abs2diff(x) = abs2.(diff(x))
abs2diff!(y, x) = y .= abs2.(diff(x))

sumabs2diff(x) = sum(abs2diff(x))

function abs2diff_jacobian(x)
    n = length(x)
    return SparseArrays.spdiagm(n - 1, n, 0 => -2 * diff(x), 1 => 2 * diff(x))
end

function sumabs2diff_hessian(x)
    n = length(x)
    return SparseArrays.spdiagm(
        0 => vcat(2.0, fill(4.0, n - 2), 2.0),
        1 => fill(-2.0, n - 1),
        -1 => fill(-2.0, n - 1),
    )
end

test_differentiation(
    sparse_backends,
    [
        JacobianScenario(abs2diff; x=rand(5), ref=abs2diff_jacobian, operator=:outofplace),
        JacobianScenario(abs2diff; x=rand(5), ref=abs2diff_jacobian, operator=:inplace),
        JacobianScenario(
            abs2diff!; x=rand(5), y=zeros(4), ref=abs2diff_jacobian, operator=:outofplace
        ),
        JacobianScenario(
            abs2diff!; x=rand(5), y=zeros(4), ref=abs2diff_jacobian, operator=:inplace
        ),
    ];
    sparsity=true,
    logging=get(ENV, "CI", "false") == "false",
)

test_differentiation(
    sparse_second_order_backends,
    [
        HessianScenario(
            sumabs2diff; x=rand(5), ref=sumabs2diff_hessian, operator=:outofplace
        ),
        HessianScenario(sumabs2diff; x=rand(5), ref=sumabs2diff_hessian, operator=:inplace),
    ];
    sparsity=true,
    logging=get(ENV, "CI", "false") == "false",
)
