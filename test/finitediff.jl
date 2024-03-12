using ADTypes: AutoFiniteDiff
using FiniteDiff: FiniteDiff

test_pushforward(AutoFiniteDiff(), scenarios; type_stability=true);
test_jacobian_and_friends(AutoFiniteDiff(), scenarios; type_stability=false);
