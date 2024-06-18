function identity_scenarios(x::Number; dx::Number, dy::Number)
    nb_args = 1
    place = :outofplace
    f = identity
    y = f(x)
    dy_from_dx = dx
    dx_from_dy = dy
    der = one(x)
    der2 = zero(x)

    # everyone out of place
    return [
        PushforwardScenario(f; x, y, dx, dy=dy_from_dx, nb_args, place),
        PullbackScenario(f; x, y, dy, dx=dx_from_dy, nb_args, place),
        DerivativeScenario(f; x, y, der, nb_args, place),
        SecondDerivativeScenario(f; x, y, der, der2, nb_args, place),
    ]
end

function copyto!_scenarios(x::AbstractArray; dx::AbstractArray, dy::AbstractArray)
    nb_args = 2
    f! = copyto!
    y = similar(x)
    f!(y, x)
    dy_from_dx = dx
    dx_from_dy = dy
    jac = Matrix(Diagonal(ones(eltype(x), length(x))))

    return all_array_to_array_scenarios(
        f!; x, y, dx, dy, dy_from_dx, dx_from_dy, jac, nb_args
    )
end

function allocfree_scenarios(rng::AbstractRNG)
    x_ = rand(rng)
    dx_ = rand(rng)
    dy_ = rand(rng)

    x_6 = rand(rng, 6)
    dx_6 = rand(rng, 6)
    dy_6 = rand(rng, 6)

    return vcat(
        identity_scenarios(x_; dx=dx_, dy=dy_), #
        copyto!_scenarios(x_6; dx=dx_6, dy=dy_6),
    )
end
