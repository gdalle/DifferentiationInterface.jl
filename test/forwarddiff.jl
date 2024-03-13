using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff

forwarddiff_backend = AutoForwardDiff(; chunksize=2)

test_pushforward(forwarddiff_backend, scenarios; type_stability=true);
test_derivative(forwarddiff_backend, scenarios; type_stability=true);
test_multiderivative(forwarddiff_backend, scenarios; type_stability=true);
test_gradient(forwarddiff_backend, scenarios; type_stability=true);
test_jacobian(forwarddiff_backend, scenarios; type_stability=false);
