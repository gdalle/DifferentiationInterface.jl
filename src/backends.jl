"""
    check_available(backend)

Check whether `backend` is available by trying a scalar-to-scalar derivative.
Might take a while due to compilation time.
"""
function check_available(backend::AbstractADType)
    try
        value_and_gradient(abs2, backend, 2.0)
        return true
    catch e
        if e isa MethodError
            return false
        else
            throw(e)
        end
    end
end

square!(y, x) = y .= x .^ 2

"""
    check_mutation(backend)

Check whether `backend` supports differentiation of mutating functions by trying a jacobian.
Might take a while due to compilation time.
"""
function check_mutation(backend::AbstractADType)
    try
        y, jac = value_and_jacobian!!(square!, [0.0], [0.0;;], backend, [3.0])
        return isapprox(y, [9.0]; rtol=1e-3) && isapprox(jac, [6.0;;]; rtol=1e-3)
    catch e
        return false
    end
end
