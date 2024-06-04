## Vector to vector

diffsquare(x::AbstractVector)::AbstractVector = diff(x) .^ 2
diffcube(x::AbstractVector)::AbstractVector = diff(x) .^ 3

function diffsquare!(y::AbstractVector, x::AbstractVector)
    x1 = @view x[1:(end - 1)]
    x2 = @view x[2:end]
    y .= x2 .- x1
    y .^= 2
    return nothing
end

function diffcube!(y::AbstractVector, x::AbstractVector)
    x1 = @view x[1:(end - 1)]
    x2 = @view x[2:end]
    y .= x2 .- x1
    y .^= 3
    return nothing
end

function diffsquare_jacobian(x)
    n = length(x)
    return spdiagm(n - 1, n, 0 => -2 * diff(x), 1 => 2 * diff(x))
end

function diffcube_jacobian(x)
    n = length(x)
    return spdiagm(n - 1, n, 0 => -3 * diff(x) .^ 2, 1 => 3 * diff(x) .^ 2)
end

function sparse_vec_to_vec_scenarios(x::AbstractVector)
    n = length(x)
    scens = AbstractScenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                JacobianScenario(diffsquare; x=x, ref=diffsquare_jacobian, place=place),
                JacobianScenario(
                    diffsquare!;
                    x=x,
                    y=similar(x, n - 1),
                    ref=diffsquare_jacobian,
                    place=place,
                ),
            ],
        )
    end
    return scens
end

## Matrix to vector

function diffsquarecube_matvec(x::AbstractMatrix)::AbstractVector
    return vcat(diffsquare(vec(x)), diffcube(vec(x)))
end

function diffsquarecube_matvec!(y::AbstractVector, x::AbstractMatrix)
    m, n = size(x)
    diffsquare!(view(y, 1:(m * n - 1)), vec(x))
    diffcube!(view(y, (m * n):(2(m * n) - 2)), vec(x))
    return nothing
end

function diffsquarecube_matvec_jacobian(x)
    return vcat(diffsquare_jacobian(vec(x)), diffcube_jacobian(vec(x)))
end

function sparse_mat_to_vec_scenarios(x::AbstractMatrix)
    m, n = size(x)
    scens = AbstractScenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                JacobianScenario(
                    diffsquarecube_matvec;
                    x=x,
                    ref=diffsquarecube_matvec_jacobian,
                    place=place,
                ),
                JacobianScenario(
                    diffsquarecube_matvec!;
                    x=x,
                    y=similar(x, 2(m * n) - 2),
                    ref=diffsquarecube_matvec_jacobian,
                    place=place,
                ),
            ],
        )
    end
    return scens
end

## Vector to matrix

diffsquarecube_vecmat(x::AbstractVector)::AbstractMatrix = hcat(diffsquare(x), diffcube(x))

function diffsquarecube_vecmat!(y::AbstractMatrix, x::AbstractVector)
    diffsquare!(view(y, :, 1), x)
    diffcube!(view(y, :, 2), x)
    return nothing
end

function diffsquarecube_vecmat_jacobian(x::AbstractVector)
    return vcat(diffsquare_jacobian(x), diffcube_jacobian(x))
end

function sparse_vec_to_mat_scenarios(x::AbstractVector)
    n = length(x)
    scens = AbstractScenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                JacobianScenario(
                    diffsquarecube_vecmat;
                    x=x,
                    ref=diffsquarecube_vecmat_jacobian,
                    place=place,
                ),
                JacobianScenario(
                    diffsquarecube_vecmat!;
                    x=x,
                    y=similar(x, n - 1, 2),
                    ref=diffsquarecube_vecmat_jacobian,
                    place=place,
                ),
            ],
        )
    end
    return scens
end

## Matrix to matrix

function diffsquarecube_matmat(x::AbstractMatrix)::AbstractMatrix
    return hcat(diffsquare(vec(x)), diffcube(vec(x)))
end

function diffsquarecube_matmat!(y::AbstractMatrix, x::AbstractMatrix)
    diffsquare!(view(y, :, 1), vec(x))
    diffcube!(view(y, :, 2), vec(x))
    return nothing
end

function diffsquarecube_matmat_jacobian(x::AbstractMatrix)
    return vcat(diffsquare_jacobian(vec(x)), diffcube_jacobian(vec(x)))
end

function sparse_mat_to_mat_scenarios(x::AbstractMatrix)
    m, n = size(x)
    scens = AbstractScenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                JacobianScenario(
                    diffsquarecube_matmat;
                    x=x,
                    ref=diffsquarecube_matmat_jacobian,
                    place=place,
                ),
                JacobianScenario(
                    diffsquarecube_matmat!;
                    x=x,
                    y=similar(x, m * n - 1, 2),
                    ref=diffsquarecube_matmat_jacobian,
                    place=place,
                ),
            ],
        )
    end
    return scens
end

## Vector to scalar

sumdiffcube(x::AbstractVector)::Number = sum(diffcube(x))

sumdiffcube_gradient(x::AbstractVector) = nothing

function sumdiffcube_hessian(x::AbstractVector)
    T = eltype(x)
    d = 6 * diff(x)
    return spdiagm(0 => vcat(d, zero(T)) + vcat(zero(T), d), 1 => -d, -1 => -d)
end

function sparse_vec_to_num_scenarios(x::AbstractVector)
    scens = AbstractScenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                HessianScenario(
                    sumdiffcube;
                    x=x,
                    ref=sumdiffcube_hessian,
                    first_order_ref=sumdiffcube_gradient,
                    place=place,
                ),
            ],
        )
    end
    return scens
end

## Matrix to scalar

sumdiffcube_mat(x::AbstractMatrix)::Number = sum(diffcube(vec(x)))

sumdiffcube_mat_gradient(x::AbstractMatrix) = nothing

function sumdiffcube_mat_hessian(x::AbstractMatrix)
    T = eltype(x)
    d = 6 * diff(vec(x))
    return spdiagm(0 => vcat(d, zero(T)) + vcat(zero(T), d), 1 => -d, -1 => -d)
end

function sparse_mat_to_num_scenarios(x::AbstractMatrix)
    scens = AbstractScenario[]
    for place in (:outofplace, :inplace)
        append!(
            scens,
            [
                HessianScenario(
                    sumdiffcube_mat;
                    x=x,
                    ref=sumdiffcube_mat_hessian,
                    first_order_ref=sumdiffcube_mat_gradient,
                    place=place,
                ),
            ],
        )
    end
    return scens
end

## Gather

"""
    sparse_scenarios(rng=Random.default_rng())

Create a vector of [`AbstractScenario`](@ref)s with sparse array types, focused on sparse Jacobians and Hessians.
"""
function sparse_scenarios(rng::AbstractRNG=default_rng())
    return vcat(
        sparse_vec_to_vec_scenarios(rand(rng, 6)),
        sparse_vec_to_mat_scenarios(rand(rng, 6)),
        sparse_mat_to_vec_scenarios(rand(rng, 2, 3)),
        sparse_mat_to_mat_scenarios(rand(rng, 2, 3)),
        sparse_vec_to_num_scenarios(rand(rng, 6)),
        sparse_mat_to_num_scenarios(rand(rng, 2, 3)),
    )
end
