# Keywords renamed for one convention: a keyword that names an
# unknown is named after its quantity (`density`, `energy`, `ions`, `hot_density`). The old
# names are still accepted, with a deprecation warning.

"""
    renamed_keyword(caller, new::Pair, old::Pair, default)

The value of a renamed keyword: the value of `new` (`name =>
value`) unless the caller gave the deprecated keyword `old`, whose value is then taken with a
deprecation warning. The deprecated keyword defaults to `nothing`, meaning not given. Giving
both, with the new one away from its `default`, is an error. A new keyword given at its
default cannot be told from one not given, so the deprecated one then wins:
`population_equations(c; density = :cold_electron_density, cold = :x)` builds the row of `:x`.
"""
function renamed_keyword(caller::Symbol, new::Pair{Symbol}, old::Pair{Symbol}, default)
    old.second === nothing && return new.second
    new.second == default ||
        throw(ArgumentError("$caller: give $(new.first) alone, $(old.first) is its deprecated name"))
    Base.depwarn("the keyword $(old.first) of $caller is deprecated, use $(new.first) (issue #185)", caller)
    return old.second
end
