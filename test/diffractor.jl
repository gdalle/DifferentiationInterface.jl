using ADTypes: AutoDiffractor
using Diffractor: Diffractor

test_pushforward(AutoDiffractor(), scenarios; type_stability=false);
test_jacobian_and_friends(AutoDiffractor(), scenarios; type_stability=false);
