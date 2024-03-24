using FiniteDifferences: FiniteDifferences, central_fdm

@test check_available(AutoFiniteDifferences(central_fdm(5, 1)))
@test !check_mutation(AutoFiniteDifferences(central_fdm(5, 1)))
@test_broken !check_hessian(AutoFiniteDifferences(central_fdm(5, 1)))

test_differentiation(AutoFiniteDifferences(central_fdm(5, 1)); second_order=false);
