using ADTypes: AutoZygote
using Zygote: Zygote

test_pullback(AutoZygote(), scenarios; type_stability=false);
test_jacobian_and_friends(AutoZygote(), scenarios; type_stability=false);
