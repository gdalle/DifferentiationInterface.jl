"""
    check_available(backend)

Check whether `backend` is available by trying a scalar-to-scalar derivative.
Might take a while due to compilation time.
"""
function check_available(backend::AbstractADType)
    try
        derivative(backend, abs2, 2.0)
        return true
    catch e
        throw(e)
        if e isa MethodError
            return false
        else
            throw(e)
        end
    end
end

available(backend::SecondOrder) = available(inner(backend)) && available(outer(backend))

square!(y::AbstractArray, x::AbstractArray) = y .= x .^ 2

"""
    check_mutation(backend)

Check whether `backend` supports differentiation of mutating functions by trying a jacobian.
Might take a while due to compilation time.
"""
function check_mutation(backend::AbstractADType)
    try
        x = [3.0]
        y = [0.0]
        jac = [0.0;;]
        value_and_jacobian!(y, jac, backend, square!, x)
        return isapprox(y, [9.0]; rtol=1e-3) && isapprox(jac, [6.0;;]; rtol=1e-3)
    catch e
        return false
    end
end

sqnorm(x::AbstractArray) = sum(abs2, x)

"""
    check_hessian(backend)

Check whether `backend` supports second order differentiation by trying a hessian.
Might take a while due to compilation time.
"""
function check_hessian(backend::AbstractADType)
    try
        x = [3.0]
        hess = hessian(backend, sqnorm, x)
        return isapprox(hess, [2.0;;]; rtol=1e-3)
    catch e
        return false
    end
end
