## Traits

for trait in (
    :check_available,
    :mutation_support,
    :pushforward_performance,
    :pullback_performance,
    :hvp_mode,
)
    @eval $trait(backend::AutoSparse) = $trait(dense_ad(backend))
end

## Operators

for op in (:pushforward, :pullback, :hvp)
    op! = Symbol(op, "!")
    valop = Symbol("value_and_", op)
    valop! = Symbol("value_and_", op, "!")
    prep = Symbol("prepare_", op)
    E = if op == :pushforward
        :PushforwardExtras
    elseif op == :pullback
        :PullbackExtras
    elseif op == :hvp
        :HVPExtras
    end

    ## One argument
    @eval begin
        $prep(f, ba::AutoSparse, x, v) = $prep(f, dense_ad(ba), x, v)
        $op(f, ba::AutoSparse, x, v, ex::$E=$prep(f, ba, x, v)) =
            $op(f, dense_ad(ba), x, v, ex)
        $valop(f, ba::AutoSparse, x, v, ex::$E=$prep(f, ba, x, v)) =
            $valop(f, dense_ad(ba), x, v, ex)
        $op!(f, res, ba::AutoSparse, x, v, ex::$E=$prep(f, ba, x, v)) =
            $op!(f, res, dense_ad(ba), x, v, ex)
        $valop!(f, res, ba::AutoSparse, x, v, ex::$E=$prep(f, ba, x, v)) =
            $valop!(f, res, dense_ad(ba), x, v, ex)
    end

    ## Two arguments
    @eval begin
        $prep(f!, y, ba::AutoSparse, x, v) = $prep(f!, y, dense_ad(ba), x, v)
        $op(f!, y, ba::AutoSparse, x, v, ex::$E=$prep(f!, y, ba, x, v)) =
            $op(f!, y, dense_ad(ba), x, v, ex)
        $valop(f!, y, ba::AutoSparse, x, v, ex::$E=$prep(f!, y, ba, x, v)) =
            $valop(f!, y, dense_ad(ba), x, v, ex)
        $op!(f!, y, res, ba::AutoSparse, x, v, ex::$E=$prep(f!, y, ba, x, v)) =
            $op!(f!, y, res, dense_ad(ba), x, v, ex)
        $valop!(f!, y, res, ba::AutoSparse, x, v, ex::$E=$prep(f!, y, ba, x, v)) =
            $valop!(f!, y, res, dense_ad(ba), x, v, ex)
    end

    ## Split
    if op == :pullback
        valop_split = Symbol("value_and_", op, "_split")
        valop!_split = Symbol("value_and_", op!, "_split")

        @eval begin
            $valop_split(f, ba::AutoSparse, x, ex::$E=$prep(f, ba, x, f(x))) =
                $valop_split(f, dense_ad(ba), x, ex)
            $valop!_split(f, ba::AutoSparse, x, ex::$E=$prep(f, ba, x, f(x))) =
                $valop!_split(f, dense_ad(ba), x, ex)
            $valop_split(f!, y, ba::AutoSparse, x, ex::$E=$prep(f, ba, x, similar(y))) =
                $valop_split(f!, y, dense_ad(ba), x, ex)
            $valop!_split(f!, y, ba::AutoSparse, x, ex::$E=$prep(f, ba, x, similar(y))) =
                $valop!_split(f!, y, dense_ad(ba), x, ex)
        end
    end
end

for op in (:derivative, :gradient, :second_derivative)
    op! = Symbol(op, "!")
    valop = Symbol("value_and_", op)
    valop! = Symbol("value_and_", op, "!")
    prep = Symbol("prepare_", op)
    E = if op == :derivative
        :DerivativeExtras
    elseif op == :gradient
        :GradientExtras
    elseif op == :second_derivative
        :SecondDerivativeExtras
    end

    ## One argument
    @eval begin
        $prep(f, ba::AutoSparse, x) = $prep(f, dense_ad(ba), x)
        $op(f, ba::AutoSparse, x, ex::$E=$prep(f, ba, x)) = $op(f, dense_ad(ba), x, ex)
        $valop(f, ba::AutoSparse, x, ex::$E=$prep(f, ba, x)) =
            $valop(f, dense_ad(ba), x, ex)
        $op!(f, res, ba::AutoSparse, x, ex::$E=$prep(f, ba, x)) =
            $op!(f, res, dense_ad(ba), x, ex)
        $valop!(f, res, ba::AutoSparse, x, ex::$E=$prep(f, ba, x)) =
            $valop!(f, res, dense_ad(ba), x, ex)
    end

    ## Two arguments
    if op in (:derivative,)
        @eval begin
            $prep(f!, y, ba::AutoSparse, x) = $prep(f!, y, dense_ad(ba), x)
            $op(f!, y, ba::AutoSparse, x, ex::$E=$prep(f!, y, ba, x)) =
                $op(f!, y, dense_ad(ba), x, ex)
            $valop(f!, y, ba::AutoSparse, x, ex::$E=$prep(f!, y, ba, x)) =
                $valop(f!, y, dense_ad(ba), x, ex)
            $op!(f!, y, res, ba::AutoSparse, x, ex::$E=$prep(f!, y, ba, x)) =
                $op!(f!, y, res, dense_ad(ba), x, ex)
            $valop!(f!, y, res, ba::AutoSparse, x, ex::$E=$prep(f!, y, ba, x)) =
                $valop!(f!, y, res, dense_ad(ba), x, ex)
        end
    end
end

## Jacobian, one argument (based on `value_and_jacobian!`)

function prepare_jacobian(f, backend::AutoSparse, x)
    return sparse_prepare_jacobian_aux(f, backend, x, pushforward_performance(backend))
end

function sparse_prepare_jacobian_aux(f, backend, x, ::PushforwardFast)
    y = f(x)
    sparsity = jacobian_sparsity(f, x, sparsity_detector(backend))
    column_colors = column_coloring(sparsity, coloring_algorithm(backend))
    column_color_groups = get_groups(column_colors)
    compressed_dx = similar(x)
    compressed_col = similar(y)
    pushforward_extras = prepare_pushforward(f, backend, x, compressed_dx)
    return (;
        sparsity,
        column_colors,
        column_color_groups,
        compressed_dx,
        compressed_col,
        pushforward_extras,
    )
end

function sparse_prepare_jacobian_aux(f, backend, x, ::PushforwardSlow)
    y = f(x)
    sparsity = jacobian_sparsity(f, x, sparsity_detector(backend))
    row_colors = row_coloring(sparsity, coloring_algorithm(backend))
    row_color_groups = get_groups(row_colors)
    compressed_dy = similar(y)
    compressed_row = similar(x)
    pullback_extras = prepare_pullback(f, backend, x, compressed_dy)
    return (;
        sparsity,
        row_colors,
        row_color_groups,
        compressed_dy,
        compressed_row,
        pullback_extras,
    )
end

function value_and_jacobian!(f, jac, backend::AutoSparse, x, extras::NamedTuple)
    return sparse_value_and_jacobian_aux!(
        f, jac, backend, x, extras, pushforward_performance(backend)
    )
end

function sparse_value_and_jacobian_aux!(f, jac, backend, x, extras, ::PushforwardFast)
    (; sparsity, column_color_groups, compressed_dx, compressed_col, pushforward_extras) =
        extras
    y = f(x)
    for group in column_color_groups
        compressed_dx .= zero(eltype(compressed_dx))
        for j in group
            compressed_dx[j] = one(eltype(compressed_dx))
        end
        pushforward!(f, compressed_col, backend, x, compressed_dx, pushforward_extras)
        @views for j in group
            nonzero_rows_j = (!iszero).(sparsity[:, j])
            copyto!(jac[nonzero_rows_j, j], compressed_col[nonzero_rows_j])
        end
    end
    return y, jac
end

function sparse_value_and_jacobian_aux!(f, jac, backend, x, extras, ::PushforwardSlow)
    (; sparsity, row_color_groups, compressed_dy, compressed_row, pullback_extras) = extras
    y = f(x)
    for group in row_color_groups
        compressed_dy .= zero(eltype(compressed_dy))
        for i in group
            compressed_dy[i] = one(eltype(compressed_dy))
        end
        pullback!(f, compressed_row, backend, x, compressed_dy, pullback_extras)
        @views for i in group
            nonzero_columns_i = (!iszero).(sparsity[i, :])
            copyto!(jac[i, nonzero_columns_i], compressed_row[nonzero_columns_i])
        end
    end
    return y, jac
end

function value_and_jacobian(f, backend::AutoSparse, x, extras::NamedTuple)
    jac = similar(extras.sparsity, eltype(x))
    return value_and_jacobian!(f, jac, backend, x, extras)
end

function jacobian!(f, jac, backend::AutoSparse, x, extras::NamedTuple)
    return value_and_jacobian!(f, jac, backend, x, extras)[2]
end

function jacobian(f, backend::AutoSparse, x, extras::NamedTuple)
    return value_and_jacobian(f, backend, x, extras)[2]
end

## Jacobian, two arguments (based on `jacobian!`)

function prepare_jacobian(f!, y, backend::AutoSparse, x)
    return sparse_prepare_jacobian_aux(f!, y, backend, x, pushforward_performance(backend))
end

function sparse_prepare_jacobian_aux(f!, y, backend, x, ::PushforwardFast)
    sparsity = jacobian_sparsity(f!, y, x, sparsity_detector(backend))
    column_colors = column_coloring(sparsity, coloring_algorithm(backend))
    column_color_groups = get_groups(column_colors)
    compressed_dx = similar(x)
    compressed_col = similar(y)
    pushforward_extras = prepare_pushforward(f!, y, backend, x, compressed_dx)
    return (;
        sparsity,
        column_colors,
        column_color_groups,
        compressed_dx,
        compressed_col,
        pushforward_extras,
    )
end

function sparse_prepare_jacobian_aux(f!, y, backend, x, ::PushforwardSlow)
    sparsity = jacobian_sparsity(f!, y, x, sparsity_detector(backend))
    row_colors = row_coloring(sparsity, coloring_algorithm(backend))
    row_color_groups = get_groups(row_colors)
    compressed_dy = similar(y)
    compressed_row = similar(x)
    pullback_extras = prepare_pullback(f!, y, backend, x, compressed_dy)
    return (;
        sparsity,
        row_colors,
        row_color_groups,
        compressed_dy,
        compressed_row,
        pullback_extras,
    )
end

function jacobian!(f!, y, jac, backend::AutoSparse, x, extras::NamedTuple)
    return sparse_jacobian_aux!(
        f!, y, jac, backend, x, extras, pushforward_performance(backend)
    )
end

function sparse_jacobian_aux!(f!, y, jac, backend, x, extras, ::PushforwardFast)
    (; sparsity, column_color_groups, compressed_dx, compressed_col, pushforward_extras) =
        extras
    for group in column_color_groups
        compressed_dx .= zero(eltype(compressed_dx))
        for j in group
            compressed_dx[j] = one(eltype(compressed_dx))
        end
        pushforward!(f!, y, compressed_col, backend, x, compressed_dx, pushforward_extras)
        @views for j in group
            nonzero_rows_j = (!iszero).(sparsity[:, j])
            copyto!(jac[nonzero_rows_j, j], compressed_col[nonzero_rows_j])
        end
    end
    return jac
end

function sparse_jacobian_aux!(f!, y, jac, backend, x, extras, ::PushforwardSlow)
    (; sparsity, row_color_groups, compressed_dy, compressed_row, pullback_extras) = extras
    for group in row_color_groups
        compressed_dy .= zero(eltype(compressed_dy))
        for i in group
            compressed_dy[i] = one(eltype(compressed_dy))
        end
        pullback!(f!, y, compressed_row, backend, x, compressed_dy, pullback_extras)
        @views for i in group
            nonzero_columns_i = (!iszero).(sparsity[i, :])
            copyto!(jac[i, nonzero_columns_i], compressed_row[nonzero_columns_i])
        end
    end
    return jac
end

function value_and_jacobian!(f!, y, jac, backend::AutoSparse, x, extras::NamedTuple)
    jacobian!(f!, y, jac, backend, x, extras)
    f!(y, x)
    return y, jac
end

function jacobian(f!, y, backend::AutoSparse, x, extras::NamedTuple)
    jac = similar(extras.sparsity, eltype(y))
    return jacobian!(f!, y, jac, backend, x, extras)
end

function value_and_jacobian(f!, y, backend::AutoSparse, x, extras::NamedTuple)
    jac = similar(extras.sparsity, eltype(y))
    return value_and_jacobian!(f!, y, jac, backend, x, extras)
end

## Hessian, one argument

function prepare_hessian(f, backend::AutoSparse, x)
    sparsity = hessian_sparsity(f, x, sparsity_detector(backend))
    symmetric_colors = symmetric_coloring(sparsity, coloring_algorithm(backend))
    symmetric_color_groups = get_groups(symmetric_colors)
    compressed_v = similar(x)
    compressed_col = similar(x)
    hvp_extras = prepare_hvp(f, backend, x, compressed_v)
    return (;
        sparsity,
        symmetric_colors,
        symmetric_color_groups,
        compressed_v,
        compressed_col,
        hvp_extras,
    )
end

function hessian!(f, hess, backend::AutoSparse, x, extras::NamedTuple)
    (; sparsity, symmetric_color_groups, compressed_v, compressed_col, hvp_extras) = extras
    for group in symmetric_color_groups
        compressed_v .= zero(eltype(compressed_v))
        for j in group
            compressed_v[j] = one(eltype(compressed_v))
        end
        hvp!(f, compressed_col, backend, x, compressed_v, hvp_extras)
        @views for j in group
            for i in axes(hess, 1)
                if (!iszero(sparsity[i, j]) && count(!iszero, sparsity[i, group]) == 1)
                    hess[i, j] = compressed_col[i]
                    hess[j, i] = compressed_col[i]
                end
            end
        end
    end
    return hess
end

function hessian(f, backend::AutoSparse, x, extras::NamedTuple)
    hess = similar(extras.sparsity, eltype(x))
    return hessian!(f, hess, backend, x, extras)
end
