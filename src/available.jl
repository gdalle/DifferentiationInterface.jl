"""
    available(backend)

Check whether `backend` is available by trying a scalar-to-scalar derivative.
Might take long due to compilation time.
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
