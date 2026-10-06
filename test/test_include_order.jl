# The include order of src/Runes2P.jl (decision of the author, 2026-10-06): no source file
# uses a name whose first definition is in a later file of the order. Julia resolves the
# names in the body of a function when it runs, so the package loads either way, and this
# test keeps the layers readable from the top down. A name counts as defined by a method,
# a type, a constant, a global assignment, a macro, a module or a `using X: name` of the
# file. Methods of another module (`Base.show`) define nothing here. A use is a read of a
# global name, `Runes2P.name` included: a name bound in an enclosing scope (an argument, a
# local variable, a loop variable) is not one.

using Test

const SOURCE = joinpath(dirname(@__DIR__), "src")

"""
    included_files(path)

The files that the module file `path` includes, in the order of its `include` calls,
relative to `src/`.
"""
function included_files(path)
    files = String[]
    walk(ex) = ex isa Expr && (ex.head === :call && ex.args[1] === :include ?
                                   push!(files, ex.args[2]) : foreach(walk, ex.args))
    walk(Meta.parseall(read(path, String); filename=path))
    return files
end

# the expression of a definition behind docstrings and macros (`@kwdef struct`, `@inline f`)
function unwrap(ex)
    while ex isa Expr && ex.head === :macrocall
        ex = ex.args[end]
    end
    return ex
end

# the call of a method signature `f(x)`, `f(x)::T`, `f(x) where T`, `(x::T)(y)`, and its
# static parameters; `nothing` for any other left side of an assignment
function signature_call(sig)
    parameters = Any[]
    while sig isa Expr && (sig.head === :where ||
                           sig.head === :(::) && length(sig.args) == 2 && sig.args[1] isa Expr)
        sig.head === :where && append!(parameters, sig.args[2:end])
        sig = sig.args[1]
    end
    return sig isa Expr && sig.head === :call ? (sig, parameters) : (nothing, parameters)
end

# the name of a signature; nothing for `Base.f(x)` and the call of an instance `(x::T)(y)`
function signature_name(sig)
    call, _ = signature_call(sig)
    call === nothing && return nothing
    name = call.args[1]
    name isa Expr && name.head === :curly && (name = name.args[1])
    return name isa Symbol ? name : nothing
end

# the name of a type head `T`, `T{S}`, `T <: A`, `T{S} <: A`
function type_name(head)
    head isa Expr && head.head === :(<:) && (head = head.args[1])
    head isa Expr && head.head === :curly && (head = head.args[1])
    return head isa Symbol ? head : nothing
end

is_method(ex) = ex isa Expr && (ex.head === :function && ex.args[1] isa Expr ||
                                ex.head === :(=) && first(signature_call(ex.args[1])) !== nothing)

"""
    defined_names(ex)

The global names that the top-level expression `ex` defines.
"""
function defined_names(ex)
    ex = unwrap(ex)
    ex isa Expr || return Symbol[]
    if ex.head === :function || is_method(ex)
        name = ex.args[1] isa Symbol ? ex.args[1] : signature_name(ex.args[1])
        return name === nothing ? Symbol[] : [name]
    elseif ex.head === :struct
        return filter(!isnothing, [type_name(ex.args[2])])
    elseif ex.head in (:abstract, :primitive)
        return filter(!isnothing, [type_name(ex.args[1])])
    elseif ex.head === :macro
        return [Symbol("@", signature_name(ex.args[1]))]
    elseif ex.head === :module
        return [ex.args[2]]
    elseif ex.head === :const
        return defined_names(ex.args[1])
    elseif ex.head === :(=)
        lhs = ex.args[1]
        lhs isa Expr && lhs.head === :(::) && (lhs = lhs.args[1])
        lhs isa Symbol && return [lhs]
        lhs isa Expr && lhs.head === :tuple && return filter(s -> s isa Symbol, lhs.args)
        return Symbol[]
    elseif ex.head in (:using, :import)
        names = Symbol[]
        for clause in ex.args
            items = clause.head === :(:) ? clause.args[2:end] : [clause]
            for item in items
                name = item.head === :as ? item.args[2] : item.args[end]
                push!(names, name isa Expr ? name.args[end] : name)
            end
        end
        return names
    elseif ex.head in (:block, :if, :elseif, :let)
        # the definitions of `@static if` and of blocks at the top level
        return reduce(vcat, (defined_names(arg) for arg in ex.args); init=Symbol[])
    end
    return Symbol[]
end

# the names that a binding form `x`, `x::T`, `(a, b)`, `x = default`, `x...` introduces
function bound!(names, ex)
    if ex isa Symbol
        push!(names, ex)
    elseif ex isa Expr
        if ex.head === :(::)
            length(ex.args) == 2 && bound!(names, ex.args[1])
        elseif ex.head in (:kw, :(=))
            bound!(names, ex.args[1])
        elseif ex.head in (:tuple, :parameters, :...)
            foreach(arg -> bound!(names, arg), ex.args)
        elseif ex.head === :call
            # a signature: its first argument is the name, not a local
            foreach(arg -> bound!(names, arg), ex.args[2:end])
        elseif ex.head === :macrocall
            bound!(names, ex.args[end])   # `@nospecialize(x)`
        end
    end
    return names
end

# the names that the body of a function assigns: its local variables, without those of the
# functions inside it
function assigned!(names, ex)
    ex isa Expr || return names
    if is_method(ex)
        name = signature_name(ex.args[1])
        name === nothing || push!(names, name)
        return names
    elseif ex.head in (:function, :->)
        ex.head === :function && ex.args[1] isa Symbol && push!(names, ex.args[1])
        return names
    elseif ex.head in (:for, :let)
        # the variables of the iteration are local to the loop: only its body assigns
        return assigned!(names, ex.args[2])
    elseif ex.head === :generator
        return assigned!(names, ex.args[1])
    elseif ex.head in (:(=), :local, :+=, :-=, :*=, :/=)
        bound!(names, ex.args[1])
    end
    foreach(arg -> assigned!(names, arg), ex.args)
    return names
end

# the names of the static parameters `T`, `T <: A` and `{T, S}`, and the expressions
# they read
function static_parameters(parameters)
    names, read = Set{Symbol}(), Any[]
    for parameter in parameters
        if parameter isa Expr && parameter.head === :braces
            inner_names, inner_read = static_parameters(parameter.args)
            union!(names, inner_names)
            append!(read, inner_read)
        elseif parameter isa Expr && parameter.head in (:<:, :>:)
            push!(names, parameter.args[1])
            push!(read, parameter.args[2])
        elseif parameter isa Symbol
            push!(names, parameter)
        end
    end
    return names, read
end

# the names of the iteration specifications of `for`, `let` and generators, and the
# expressions they read in the scope around them
function iteration_bindings(specs)
    names, read = Set{Symbol}(), Any[]
    for spec in specs
        spec isa Expr && spec.head === :filter && continue
        if spec isa Expr && spec.head === :(=)
            bound!(names, spec.args[1])
            push!(read, spec.args[2])
        else
            bound!(names, spec)
        end
    end
    return names, read
end

"""
    free_names!(names, ex, bound)

Add to `names` every global name that `ex` reads, given the names `bound` by the scopes
around it.
"""
function free_names!(names, ex, bound)
    if ex isa Symbol
        ex in bound || push!(names, ex)
        return names
    end
    ex isa Expr || return names
    head = ex.head
    head in (:quote, :inert, :meta, :line, :export, :public, :using, :import, :local, :global) && return names
    if is_method(ex) || head === :->
        head === :function && ex.args[1] isa Symbol && return names   # `function f end`
        call, parameters = head === :-> ? (ex.args[1], Any[]) : signature_call(ex.args[1])
        where_names, where_read = static_parameters(parameters)
        inner = union(bound, where_names, bound!(Set{Symbol}(), call))
        length(ex.args) > 1 && assigned!(inner, ex.args[2])
        # the types and defaults of the arguments, the return type and the bounds of the
        # static parameters are read; the name of the method is not
        is_call = call isa Expr && call.head === :call
        foreach(arg -> free_names!(names, arg, inner), is_call ? call.args[2:end] : [call])
        is_call && call.args[1] isa Expr && call.args[1].head === :(::) &&
            free_names!(names, call.args[1], inner)   # the type of the instance of `(x::T)(y)`
        foreach(arg -> free_names!(names, arg, bound), where_read)
        sig = head === :-> ? nothing : ex.args[1]
        while sig isa Expr && sig.head in (:where, :(::)) && length(sig.args) >= 2
            sig.head === :(::) && free_names!(names, sig.args[2], inner)
            sig = sig.args[1]
        end
        length(ex.args) > 1 && free_names!(names, ex.args[2], inner)
    elseif head === :for || head === :let
        spec = ex.args[1]
        loop_names, read = iteration_bindings(spec isa Expr && spec.head === :block ? spec.args : [spec])
        foreach(arg -> free_names!(names, arg, bound), read)
        inner = union(bound, loop_names)
        head === :let && assigned!(inner, ex.args[2])
        free_names!(names, ex.args[2], inner)
    elseif head === :generator
        loop_names, read = iteration_bindings(ex.args[2:end])
        foreach(arg -> free_names!(names, arg, bound), read)
        inner = union(bound, loop_names)
        free_names!(names, ex.args[1], inner)
        for spec in ex.args[2:end]
            spec isa Expr && spec.head === :filter && free_names!(names, spec.args[1], inner)
        end
    elseif head === :try
        free_names!(names, ex.args[1], bound)
        caught = ex.args[2] isa Symbol ? union(bound, [ex.args[2]]) : bound
        length(ex.args) >= 3 && free_names!(names, ex.args[3], caught)
        foreach(arg -> free_names!(names, arg, bound), ex.args[4:end])
    elseif head === :.
        free_names!(names, ex.args[1], bound)
        member = ex.args[2]
        # a name of the package itself, `Runes2P.name`
        ex.args[1] === :Runes2P && member isa QuoteNode && member.value isa Symbol && push!(names, member.value)
        member isa Expr && member.head === :tuple && free_names!(names, member, bound)   # broadcast
    elseif head === :macrocall
        ex.args[1] isa Symbol && push!(names, ex.args[1])
        foreach(arg -> free_names!(names, arg, bound), ex.args[2:end])
    elseif head in (:kw, :(=), :+=, :-=, :*=, :/=)
        # the target of an assignment and a keyword or field name are not reads
        lhs = ex.args[1]
        if lhs isa Expr && lhs.head in (:tuple, :(::))
            foreach(arg -> arg isa Symbol || free_names!(names, arg, bound), lhs.args)
        elseif !(lhs isa Symbol)
            free_names!(names, lhs, bound)
        end
        free_names!(names, ex.args[2], bound)
    elseif head === :struct
        type_head = ex.args[2] isa Expr && ex.args[2].head === :(<:) ? ex.args[2].args[1] : ex.args[2]
        type_parameters = type_head isa Expr && type_head.head === :curly ?
                          first(static_parameters(type_head.args[2:end])) : Set{Symbol}()
        inner = union(bound, type_parameters)
        ex.args[2] isa Expr && free_names!(names, ex.args[2], inner)
        for field in ex.args[3].args
            # a field `name`, `name::T` or `name::T = default` (`@kwdef`): its name is not a read
            if field isa Expr && field.head === :(=) && !is_method(field)
                field.args[1] isa Expr && free_names!(names, field.args[1].args[end], inner)
                free_names!(names, field.args[2], inner)
            elseif field isa Expr && field.head === :(::)
                free_names!(names, field.args[end], inner)
            elseif !(field isa Symbol)
                free_names!(names, field, inner)
            end
        end
    else
        foreach(arg -> free_names!(names, arg, bound), ex.args)
    end
    return names
end

"""
    include_order_violations(source)

The uses of a name before the file of its first definition, as tuples
`(file, line, name, defining_file)`, for the module file `source/Runes2P.jl`.
"""
function include_order_violations(source)
    files = included_files(joinpath(source, "Runes2P.jl"))
    parsed = [Meta.parseall(read(joinpath(source, file), String); filename=file) for file in files]
    first_file = Dict{Symbol,Int}()
    for (index, tree) in enumerate(parsed), ex in tree.args
        for name in defined_names(ex)
            get!(first_file, name, index)
        end
    end
    violations = Tuple{String,Int,Symbol,String}[]
    for (index, tree) in enumerate(parsed)
        line = 0
        for ex in tree.args
            ex isa LineNumberNode && (line = ex.line; continue)
            for name in sort!(collect(free_names!(Set{Symbol}(), ex, Set{Symbol}())))
                later = get(first_file, name, 0)
                later > index && push!(violations, (files[index], line, name, files[later]))
            end
        end
    end
    return violations
end

# a module of two files for the check of the checker: a.jl reads the names `later_*` and
# `Later*` that b.jl defines, and reads `g`, `later_field` and `declared` in ways that are
# no violation (an argument, a field name, a function declared in a.jl)
const EARLIER_FILE = """
f(x) = later_call(x) + 1
h(g) = g(1)
struct S
    later_field::Int
end
p(x::LaterType) = x
q(x::T) where {T<:LaterSuper} = x
r(; option=LaterDefault()) = option
Base.@kwdef struct K
    solver::Any = LaterKwdef()
end
const C = later_const()
"A documented method."
documented(x) = later_documented(x)
function loop(u)
    for later_loop in 1:2
    end
    return later_loop(u)
end
qualified() = Runes2P.later_qualified()
module_value() = LaterModule.x
function declared end
uses_declared() = declared(1)
"""

const LATER_FILE = """
later_call(x) = x
g(x) = x
later_field() = 1
struct LaterType end
abstract type LaterSuper end
struct LaterDefault end
struct LaterKwdef end
later_const() = 1
later_documented(x) = x
later_loop(u) = u
later_qualified() = 1
module LaterModule
x = 1
end
declared(x) = x
"""

@testset "include order" begin
    mktempdir() do source
        write(joinpath(source, "Runes2P.jl"), "module Runes2P\ninclude(\"a.jl\")\ninclude(\"b.jl\")\nend\n")
        write(joinpath(source, "a.jl"), EARLIER_FILE)
        write(joinpath(source, "b.jl"), LATER_FILE)
        violations = include_order_violations(source)
        @test Set(name for (_, _, name, _) in violations) ==
              Set([:later_call, :LaterType, :LaterSuper, :LaterDefault, :LaterKwdef, :later_const,
                   :later_documented, :later_loop, :later_qualified, :LaterModule])
        @test all(v -> v[1] == "a.jl" && v[4] == "b.jl", violations)
        @test ("a.jl", 1, :later_call, "b.jl") in violations
    end

    @test length(included_files(joinpath(SOURCE, "Runes2P.jl"))) > 50
    violations = include_order_violations(SOURCE)
    for (file, line, name, later) in violations
        @info "`$name` is used in $file:$line and first defined in the later file $later"
    end
    @test isempty(violations)
end
