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
