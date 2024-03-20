"""
    available(backend)

Check whether `backend` is available by trying a scalar-to-scalar derivative.
Might take a while due to compilation time.
"""
function available(backend::AbstractADType)
    try
        derivative(backend, identity, 1.0)
        return true
    catch e
        if e isa MethodError
            return false
        else
            throw(e)
        end
    end
end

available(backend::SecondOrder) = available(inner(backend)) && available(outer(backend))

"""
    supports_mutation(backend)

Check whether `backend` supports differentiation of mutating functions by trying a jacobian.
Might take a while due to compilation time.
"""
function supports_mutation(backend::AbstractADType)
    try
        value_and_jacobian!([0.0], [0.0;;], backend, copyto!, [1.0])
        return true
    catch e
        return false
    end
end

"""
    supports_hessian(backend)

Check whether `backend` supports second order differentiation by trying a hessian.
Might take a while due to compilation time.
"""
function supports_hessian(backend::AbstractADType)
    try
        hessian(backend, sum, [1.0])
        return true
    catch e
        return false
    end
end
