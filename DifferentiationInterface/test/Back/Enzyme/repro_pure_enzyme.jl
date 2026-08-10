using Enzyme

# SCRATCH REPRO BRANCH (not meant to be merged): minimal pure-Enzyme
# reproducer for the EnzymeNoDerivativeError seen on Julia 1.12 x86_64 CI in
# https://github.com/JuliaDiff/DifferentiationInterface.jl/actions/runs/31380583920/job/93429638474
#
# Reproduces DifferentiationInterface.jl's `gradient_and_hvp` for
# SecondOrder(AutoEnzyme(mode=set_runtime_activity(Forward)), AutoEnzyme(mode=set_runtime_activity(Reverse)))
# applied to a "cachified" array-to-number scenario, i.e. a function that
# stores its primal output into a user-provided cache buffer before
# returning it (DifferentiationInterfaceTest's `StoreInCache` wrapper around
# `arr_to_num_linalg`).
#
# Mirrors, without any DifferentiationInterface/DifferentiationInterfaceTest
# code:
#   - DifferentiationInterface/src/second_order/hvp.jl: `gradient_and_hvp`
#     (ForwardOverAnythingHVPPrep method), which pushes forward
#     `shuffled_gradient` with the outer (forward) backend.
#   - DifferentiationInterface/src/first_order/gradient.jl: `shuffled_gradient`,
#     which calls `gradient` with the inner (reverse) backend.
#   - DifferentiationInterface/ext/DifferentiationInterfaceEnzymeExt/reverse_onearg.jl:
#     `DI.gradient` (generic contexts method), which calls
#     `Enzyme.autodiff(mode, Const(f), Active, Duplicated(x, grad), Duplicated(cache, dcache))`.
#   - the `Cache` context is always translated to `Duplicated` on the Enzyme side
#     (ext/.../utils.jl `_translate_prepared!(dc, ::DI.Cache, ::Val{1})`).
#
# NOTE: this does NOT reproduce on aarch64 (ran clean, correct values,
# 2026-08-10). It is only known to fail on the ubuntu-latest (x86_64) CI
# runner, so this script does not need to run cleanly everywhere to be a
# valid reproducer of the CI bug.

const α = 4
const β = 6

arr_to_num_linalg(x::AbstractArray) = sum(vec(x .^ α) .* transpose(vec(x .^ β)))

# DifferentiationInterfaceTest.StoreInCache{:out}(f)(x, y_cache)
function store_in_cache(x, y_cache)
    y = arr_to_num_linalg(x)
    y_cache[1] = y
    return y_cache[1]
end

# inner: reverse-mode gradient of `store_in_cache` w.r.t. x, with y_cache as Duplicated
function inner_gradient(x, y_cache)
    grad = zero(x)
    dcache = Enzyme.make_zero(y_cache)
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Const(store_in_cache),
        Active,
        Duplicated(x, grad),
        Duplicated(y_cache, dcache),
    )
    return grad
end

# outer: forward-mode pushforward of inner_gradient (forward-over-reverse HVP)
function gradient_and_hvp(x, tx, y_cache)
    dcache_outer = Enzyme.make_zero(y_cache)
    mode = Enzyme.WithPrimal(Enzyme.set_runtime_activity(Enzyme.Forward))
    hv, grad = Enzyme.autodiff(
        mode,
        inner_gradient,
        Duplicated(x, tx),
        Duplicated(y_cache, dcache_outer),
    )
    return grad, hv
end

x = collect(1.0:6.0)
tx = fill(1.0, 6)
y_cache = zeros(1)

grad, hv = gradient_and_hvp(x, tx, y_cache)
println("grad = ", grad)
println("hv   = ", hv)
