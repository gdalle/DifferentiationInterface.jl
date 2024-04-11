"""
    check_available(backend)

Check whether `backend` is available by trying a gradient.

!!! warning
    Might take a while due to compilation time.
"""
function check_available(backend::AbstractADType)
    try
        value_and_gradient(sum, backend, [1.0])
        return true
    catch exception
        @warn "Backend $backend not available" exception
        if exception isa MethodError
            return false
        else
            throw(exception)
        end
    end
end

function square!(y, x)
    y .= x .^ 2
    return nothing
end

"""
    check_mutation(backend)

Check whether `backend` supports differentiation of mutating functions by trying a jacobian.

!!! warning
    Might take a while due to compilation time.
"""
function check_mutation(backend::AbstractADType)
    try
        y, jac = value_and_jacobian!(square!, [0.0], [0.0;;], backend, [3.0])
        return isapprox(y, [9.0]; rtol=1e-3) && isapprox(jac, [6.0;;]; rtol=1e-3)
    catch exception
        @warn "Backend $backend does not support mutation" exception
        return false
    end
end

sqnorm(x::AbstractArray) = sum(abs2, x)

"""
    check_hessian(backend)

Check whether `backend` supports second order differentiation by trying a hessian.

!!! warning
    Might take a while due to compilation time.
"""
function check_hessian(backend::AbstractADType)
    try
        x = [3.0]
        hess = hessian(sqnorm, backend, x)
        return isapprox(hess, [2.0;;]; rtol=1e-3)
    catch exception
        @warn "Backend $backend does not support hessian" exception
        return false
    end
end
