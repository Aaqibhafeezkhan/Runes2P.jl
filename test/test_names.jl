# The names that users of Runes2P see are ASCII: the exported and public names, the keyword
# arguments of their methods and the fields of their types (the section "Names" of
# CONTRIBUTING.md). A local variable may still use the Greek symbol of a cited formula, so
# the source files are not scanned.
using Test
using Runes2P

ascii_name(name) = all(isascii, String(name))

# the exported and public names of Runes2P and their objects
function public_objects()
    objects = Pair{Symbol, Any}[]
    for name in names(Runes2P)
        name === :Runes2P && continue
        push!(objects, name => getfield(Runes2P, name))
    end
    return objects
end

# the methods of `f` defined in Runes2P, its submodules or its extensions (when loaded)
function own_methods(f)
    root(m) = String(nameof(Base.moduleroot(parentmodule(m))))
    return [m for m in methods(f) if startswith(root(m), "Runes2P")]
end

@testset "names that users see" begin
    objects = public_objects()
    @test !isempty(objects)

    @testset "exported and public names" begin
        bad = [name for (name, _) in objects if !ascii_name(name)]
        @test isempty(bad)
    end

    @testset "keyword arguments" begin
        bad = Tuple{Symbol, Symbol}[]
        for (name, object) in objects
            (object isa Function || object isa Type) || continue
            for m in own_methods(object), keyword in Base.kwarg_decl(m)

                ascii_name(keyword) || push!(bad, (name, keyword))
            end
        end
        @test isempty(bad)
    end

    @testset "fields of the types" begin
        bad = Tuple{Symbol, Symbol}[]
        for (name, object) in objects
            object isa Type || continue
            type = Base.unwrap_unionall(object)
            (type isa DataType && !isabstracttype(type)) || continue
            for field in fieldnames(type)
                ascii_name(field) || push!(bad, (name, field))
            end
        end
        @test isempty(bad)
    end
end
