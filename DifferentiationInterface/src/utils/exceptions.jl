struct MissingBackendError <: Exception
    backend::AbstractADType
end

function Base.showerror(io::IO, e::MissingBackendError)
    println(io, "MissingBackendError: Failed to use $(e.backend).")
    if !check_available(e.backend)
        print(
            io,
            """Backend package is probably not loaded. To fix this, try to run

                import $(package_name(e.backend))
            """,
        )
    else
        print(
            io,
            "Please open an issue: https://github.com/gdalle/DifferentiationInterface.jl/issues/new",
        )
    end
end
