function identity_scenarios(x::Number; dx::Number, dy::Number)
    f = identity
    y = f(x)
    dy_from_dx = dx
    dx_from_dy = dy
    der = one(x)

    return [
        Scenario{:pushforward,:out}(f, x; tang=Tangents(dx), res1=Tangents(dy_from_dx)),
        Scenario{:pullback,:out}(f, x; tang=Tangents(dy), res1=Tangents(dx_from_dy)),
        Scenario{:derivative,:out}(f, x; res1=der),
    ]
end

function sum_scenarios(x::AbstractArray; dx::AbstractArray, dy::Number)
    f = sum
    y = f(x)
    dy_from_dx = sum(dx)
    dx_from_dy = (similar(x) .= dy)
    grad = similar(x)
    grad .= one(eltype(x))

    return [
        Scenario{:pushforward,:out}(f, x; tang=Tangents(dx), res1=Tangents(dy_from_dx)),
        Scenario{:pullback,:in}(f, x; tang=Tangents(dy), res1=Tangents(dx_from_dy)),
        Scenario{:gradient,:in}(f, x; res1=grad),
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
        Scenario{:pushforward,:in}(f!, y, x; tang=Tangents(dx), res1=Tangents(dy_from_dx)),
        Scenario{:pullback,:in}(f!, y, x; tang=Tangents(dy), res1=Tangents(dx_from_dy)),
        Scenario{:jacobian,:in}(f!, y, x; res1=jac),
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
    return scens
end
