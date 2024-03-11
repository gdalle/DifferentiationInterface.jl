abstract type AbstractMode end

"""
    ForwardMode

Trait identifying forward mode AD backends.
Used for internal dispatch only.
"""
struct ForwardMode <: AbstractMode end

"""
    ReverseMode

Trait identifying reverse mode AD backends.
Used for internal dispatch only.
"""
struct ReverseMode <: AbstractMode end
