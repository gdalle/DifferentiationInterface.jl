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

hess_checker(x::AbstractArray) = x[1] * x[1] * x[2] * x[2]

"""
    check_hessian(backend)

Check whether `backend` supports second order differentiation by trying to compute a hessian.

!!! warning
    Might take a while due to compilation time.
"""
function check_hessian(backend::AbstractADType; verbose=true)
    try
        x = [2.0, 3.0]
        hess = hessian(hess_checker, backend, x)
        hess_th = [
            2*abs2(x[2]) 4*x[1]*x[2]
            4*x[1]*x[2] 2*abs2(x[1])
        ]
        return isapprox(hess, hess_th; rtol=1e-3)
    catch exception
        verbose && @warn "Backend $backend does not support hessian" exception
        return false
    end
end
