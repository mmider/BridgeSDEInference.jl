mutable struct foo
    bar
    foobar
    foo(a) = new(a)
end

f = foo([1.0])
eltype(f.bar)
f.bar = [2]
f.bar
eltype(f.bar)


foo(10)

function bar(a::foo)
    a.foobar = 100
end

function goo(b)
    b = 200
end

temp = foo(10)

bar(temp)

temp

goo(temp.foobar)

temp



struct Foo
    a::Int64
    b::Int64
end

next(f::Foo) = Foo(f.a+1, f.b+1)

mutable struct Bar
    a::Int64
    b::Int64
end

function next!(b::Bar)
    b.a += 1
    b.b += 1
end


function foo()
    f = Foo(1,2)
    for i in 1:1000
        f = next(f)
    end
    print(f)
end

function bar()
    b = Bar(1,2)
    for i in 1:1000
        next!(b)
    end
    print(b)
end

@time foo()

@time bar()

foo(::Val{false}=Val{false}()) = print("false...\n")
foo(::Val{true}=Val{false}()) = print("true...\n")

foo()

using StaticArrays
prog = "SMatrix{2,2}([ 1.0 1.0; 2.0 2.0])"
ex1 = Meta.parse(prog)
ex1.args
ex2 = Expr(:call, SMatrix{2,2}, [1.0 1.0; 2.0 2.0])
dump(ex2)


@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = tuplejoin(tuplejoin(x, y), z...)

@generated function φ(::Val{T}, args...) where T
    data = Expr(:call, :tuplejoin, (:(phi(Val($i), args...)) for i in 1:length(T) if T[i])...)
    mat = Expr(:call, SMatrix{num_non_hypo(args[3]),sum(T)}, data)
    return mat
end

@generated function



@generated function φ(::Val{T}, args...) where T
    z = Expr(:call, SMatrix{num_non_hypo(args[3]),sum(T)}, (:jointuple, (:(phi(Val($i), args...)) for i in 1:length(T) if T[i])...))
    return z
end
@SVector[1.0, 2.0, 3.0]
φ(Val{(true, true, true)}(), 10.0, @SVector[1.0,2.0], P˟)
φᶜlinear(Val{(true, true, true)}(),10.0, @SVector[1.0,2.0], P˟)
φᶜtemp(Val{(true, true, true)}(),@SVector[1.0], 10.0, @SVector[1.0,2.0], P˟)
foo_1() = (1,2)
foo_2() = (2,3)
SMatrix{2,2}((foo_1(), foo_2()))


function foo(args)
    print(args[2], "\n")
end
@generated function foo(P)
    z = Expr(num_non_hypo(P))
    return z
end
param = :complexConjug
θ_init = [10.0, -8.0, 15.0, 0.0, 3.0]
P˟ = FitzhughDiffusion(param, θ_init...)
foo(P˟)
num_non_hypo(P˟)


SMatrix{2,2}(collect(Iterators.flatten(((1,2),(4,5)))))

Expr(:call, SMatrix{2,2}, [2 3;4 5])
for i in Iterators.flatten(((1,2),(2,3))) print(i, "\n") end

((1,2),(2,3))
