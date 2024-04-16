## Vector to vector

abs2diff(x) = abs2.(diff(x))

function abs2diff!(y, x)
    y .= abs2.(diff(x))
    return nothing
end

function abs2diff_jacobian(x)
    n = length(x)
    return spdiagm(n - 1, n, 0 => -2 * diff(x), 1 => 2 * diff(x))
end

## Vector to scalar

sumabs2diff(x) = sum(abs2diff(x))

function sumabs2diff_hessian(x)
    T = eltype(x)
    n = length(x)
    return spdiagm(
        0 => vcat(2 * one(T), fill(4 * one(T), n - 2), 2 * one(T)),
        1 => fill(-2 * one(T), n - 1),
        -1 => fill(-2 * one(T), n - 1),
    )
end

## Gather

"""
    sparse_scenarios()

Create a vector of [`AbstractScenario`](@ref)s with sparse array types, focused on sparse Jacobians and Hessians.
"""
function sparse_scenarios(x::AbstractVector)
    n = length(x)
    scens = AbstractScenario[]
    for op in (:outofplace, :inplace)
        append!(
            scens,
            [
                JacobianScenario(abs2diff; x=x, ref=abs2diff_jacobian, operator=op),
                JacobianScenario(
                    abs2diff!; x=x, y=similar(x, n - 1), ref=abs2diff_jacobian, operator=op
                ),
                HessianScenario(sumabs2diff; x=x, ref=sumabs2diff_hessian, operator=op),
            ],
        )
    end
    return scens
end
