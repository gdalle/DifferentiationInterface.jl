"""
    AutoBrokenForward <: ADTypes.AbstractADType

Available forward-mode backend with no pushforward implementation.
Used to test error messages.
"""
struct AutoBrokenForward <: AbstractADType end
ADTypes.mode(::AutoBrokenForward) = ADTypes.ForwardMode()
DifferentiationInterface.check_available(::AutoBrokenForward) = true

"""
    AutoBrokenReverse <: ADTypes.AbstractADType

Available reverse-mode backend with no pullback implementation.
Used to test error messages. 
"""
struct AutoBrokenReverse <: AbstractADType end
ADTypes.mode(::AutoBrokenReverse) = ADTypes.ReverseMode()
DifferentiationInterface.check_available(::AutoBrokenReverse) = true

## Test exceptions
@testset "MissingBackendError" begin
    f(x::AbstractArray) = sum(abs2, x)
    x = [1.0, 2.0, 3.0]

    @test_throws DI.MissingBackendError gradient(f, AutoBrokenForward(), x)
    @test_throws DI.MissingBackendError gradient(f, AutoBrokenReverse(), x)

    @test_throws DI.MissingBackendError hvp(f, AutoBrokenForward(), x, x)
    @test_throws DI.MissingBackendError hvp(f, AutoBrokenReverse(), x, x)
end
