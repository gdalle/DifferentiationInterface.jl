"""
    check_available(backend)

Check whether `backend` is available (i.e. whether the extension is loaded).
"""
check_available(backend::AbstractADType) = false

function check_available(backend::SecondOrder)
    return check_available(inner(backend)) && check_available(outer(backend))
end

"""
    check_twoarg(backend)

Check whether `backend` supports differentiation of two-argument functions.
"""
check_twoarg(backend::AbstractADType) = Bool(twoarg_support(backend))

sqnorm(x::AbstractArray) = sum(abs2.(x))

"""
    check_hessian(backend)

Check whether `backend` supports second order differentiation by trying to compute a hessian.

!!! warning
    Might take a while due to compilation time.
"""
function check_hessian(backend::AbstractADType; verbose=true)
    try
        x = [1.0, 3.0]
        hess = hessian(sqnorm, backend, x)
        return isapprox(hess, [2.0 0.0; 0.0 2.0]; rtol=1e-3)
    catch exception
        if verbose
            @warn "Backend $backend does not support hessian" exception
        end
        return false
    end
end
