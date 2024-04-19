using ADTypes
using DifferentiationInterface
using ForwardDiff: ForwardDiff
using Symbolics: Symbolics
using Zygote: Zygote

ca = DifferentiationInterface.GreedyColoringAlgorithm()
symbolics_ext = Base.get_extension(
    DifferentiationInterface, :DifferentiationInterfaceSymbolicsExt
)
sd = symbolics_ext.SymbolicsSparsityDetector()

sparse_backend_forward = AutoSparse(
    AutoForwardDiff(); sparsity_detector=sd, coloring_algorithm=ca
)
sparse_backend_reverse = AutoSparse(
    AutoEnzyme(Enzyme.Reverse); sparsity_detector=sd, coloring_algorithm=ca
)

sparse_second_order_forward = AutoSparse(
    SecondOrder(AutoForwardDiff(), AutoForwardDiff());
    sparsity_detector=sd,
    coloring_algorithm=ca,
)
sparse_second_order_reverse = AutoSparse(
    SecondOrder(AutoZygote(), AutoForwardDiff());
    sparsity_detector=sd,
    coloring_algorithm=ca,
)

x = rand(3, 3)
y = rand(4, 2)
f(x) = reshape((diff(vec(x))) .^ 3, (4, 2))
f!(y, x) = y .= f(x)

ADTypes.jacobian_sparsity(f, x, sd)
ADTypes.jacobian_sparsity(f!, y, x, sd)

jacobian(f, sparse_backend_forward, x)
jacobian(f, sparse_backend_reverse, x)

jacobian(f!, y, sparse_backend_forward, x)
jacobian(f!, y, sparse_backend_reverse, x)

g(x) = sum(f(x))

ADTypes.hessian_sparsity(g, x, sd)

hessian(g, sparse_second_order_forward, x)

hessian(g, AutoForwardDiff(), x)
