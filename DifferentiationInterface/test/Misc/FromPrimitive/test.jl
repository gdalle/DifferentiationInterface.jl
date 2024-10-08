using DifferentiationInterface, DifferentiationInterfaceTest
using DifferentiationInterface: AutoForwardFromPrimitive, AutoReverseFromPrimitive
using DifferentiationInterfaceTest
using ForwardDiff: ForwardDiff
using Test

LOGGING = get(ENV, "CI", "false") == "false"

backends = [ #
    AutoForwardFromPrimitive(AutoForwardDiff(; chunksize=5)),
    AutoReverseFromPrimitive(AutoForwardDiff(; chunksize=5)),
]

second_order_backends = [ #
    SecondOrder(
        AutoForwardFromPrimitive(AutoForwardDiff(; chunksize=5)),
        AutoReverseFromPrimitive(AutoForwardDiff(; chunksize=5)),
    ),
    SecondOrder(
        AutoReverseFromPrimitive(AutoForwardDiff(; chunksize=5)),
        AutoForwardFromPrimitive(AutoForwardDiff(; chunksize=5)),
    ),
]

for backend in vcat(backends, second_order_backends)
    @test check_available(backend)
    @test check_inplace(backend)
end

## Dense scenarios

test_differentiation(
    vcat(backends, second_order_backends),
    default_scenarios(; include_constantified=true);
    logging=LOGGING,
);

## Sparse scenarios

test_differentiation(
    MyAutoSparse.(vcat(backends, second_order_backends)),
    default_scenarios(; include_constantified=true);
    logging=LOGGING,
);

test_differentiation(
    MyAutoSparse.(vcat(backends, second_order_backends)),
    sparse_scenarios(; include_constantified=true);
    sparsity=true,
    logging=LOGGING,
);

## Misc

jac_for_prep = prepare_jacobian(copy, MyAutoSparse(backends[1]), rand(10));
jac_rev_prep = prepare_jacobian(copy, MyAutoSparse(backends[2]), rand(10));
hess_prep = prepare_hessian(x -> sum(abs2, x), MyAutoSparse(backends[1]), rand(10));

@test all(==(1), column_colors(jac_for_prep))
@test all(==(1), row_colors(jac_rev_prep))
@test all(==(1), column_colors(hess_prep))
@test only(column_groups(jac_for_prep)) == 1:10
@test only(row_groups(jac_rev_prep)) == 1:10
@test only(column_groups(hess_prep)) == 1:10
