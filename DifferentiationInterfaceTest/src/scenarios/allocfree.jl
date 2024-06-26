function identity_scenarios(x::Number; dx::Number, dy::Number)
    nb_args = 1
    place = :outofplace
    f = identity
    y = f(x)
    dy_from_dx = dx
    dx_from_dy = dy
    der = one(x)

    return [
        PushforwardScenario(f; x, y, dx, dy=dy_from_dx, nb_args, place),
        PullbackScenario(f; x, y, dy, dx=dx_from_dy, nb_args, place),
        DerivativeScenario(f; x, y, der, nb_args, place),
    ]
end

function sum_scenarios(x::AbstractArray; dx::AbstractArray, dy::Number)
    nb_args = 1
    f = sum
    y = f(x)
    dy_from_dx = sum(dx)
    dx_from_dy = (similar(x) .= dy)
    grad = x

    return [
        PushforwardScenario(f; x, y, dx, dy=dy_from_dx, nb_args, place=:outofplace),
        PullbackScenario(f; x, y, dy, dx=dx_from_dy, nb_args, place=:inplace),
        GradientScenario(f; x, y, grad, nb_args, place=:inplace),
    ]
end

function copyto!_scenarios(x::AbstractArray; dx::AbstractArray, dy::AbstractArray)
    nb_args = 2
    place = :inplace
    f! = copyto!
    y = similar(x)
    f!(y, x)
    dy_from_dx = dx
    dx_from_dy = dy
    jac = Matrix(Diagonal(ones(eltype(x), length(x))))

    return [
        PushforwardScenario(f!; x, y, dx, dy=dy_from_dx, nb_args, place),
        PullbackScenario(f!; x, y, dy, dx=dx_from_dy, nb_args, place),
        JacobianScenario(f!; x, y, jac, nb_args, place),
    ]
end

"""
    allocfree_scenarios(rng::AbstractRNG=default_rng())

Create a vector of [`Scenario`](@ref)s with functions that do not allocate.

!!! warning
    At the moment, some second-order scenarios are excluded.
"""
function allocfree_scenarios(rng::AbstractRNG=default_rng())
    x_ = rand(rng)
    dx_ = rand(rng)
    dy_ = rand(rng)

    x_6 = rand(rng, 6)
    dx_6 = rand(rng, 6)
    dy_6 = rand(rng, 6)

    scens = vcat(
        identity_scenarios(x_; dx=dx_, dy=dy_), #
        sum_scenarios(x_6; dx=dx_6, dy=dy_),
        copyto!_scenarios(x_6; dx=dx_6, dy=dy_6),
    )
    add_batchified!(scens)
    return scens
end
