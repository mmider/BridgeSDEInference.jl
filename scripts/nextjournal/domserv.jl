using JSServe, WGLMakie, AbstractPlotting
using JSServe: JSServe.DOM, @js_str, onjs
global three, scene

using StaticArrays
using Colors
using Random
rebirth(α, R) = x -> (rand() > α  ? x : (2rand(typeof(x)) .- 1).*R)
const 𝕏 = SVector



using Hyperscript, Markdown
using JSServe, Observables
using JSServe: Application, Session, evaljs, linkjs, div, active_sessions, Asset
using JSServe: @js_str, onjs, Button, TextField, Slider, JSString, Dependency, with_session
using JSServe.DOM


function dom_handler(session, request)
    global three, scene

    # slider and field for sigma
    sliders = JSServe.Slider(0.01:0.01:1)
    nrs = JSServe.NumberInput(0.0)
    linkjs(session, sliders.value, nrs.value)

    # time wheel ;-)
    button = JSServe.Slider(1:109)

    # init
    R = 𝕏(1.5,6.0)
    R1, R2 = R
    limits = FRect(-R[1], -R[2], 2R[1], 2R[2])
    n = 800
    K = 80
    dt = 0.001
    sqrtdt = sqrt(dt)

    particlecss = Asset(joinpath(@__DIR__,"particle.css"))
    ms = 0.03
    global scene = scatter(repeat(2randn(n), outer=K), repeat(2randn(n),outer=K), color = fill(:white, n*K),
        backgroundcolor = RGB{Float32}(0.04, 0.11, 0.22), markersize = ms,
        glowwidth = 0.005, glowcolor = :white,
        resolution=(600,600), limits = limits,
        )
    axis = scene[Axis]
    axis[:grid, :linewidth] =  (0.3, 0.3)
    axis[:grid, :linecolor] = (RGBA{Float32}(0.5, 0.7, 1.0, 0.3),RGBA{Float32}(0.5, 0.7, 1.0, 0.3))
    axis[:names][:textsize] = (0.0,0.0)
    axis[:ticks, :textcolor] = (RGBA{Float32}(0.5, 0.7, 1.0, 0.5),RGBA{Float32}(0.5, 0.7, 1.0, 0.5))


    splot = scene[end]
    scatter!(scene, -R1:0.01:R1, sin.(-R1:0.01:R1), color = RGBA{Float32}(0.5, 0.7, 1.0, 0.8), markersize=ms)
    kplot = scene[end]

    three, canvas = WGLMakie.three_display(session, scene)
    js_scene = WGLMakie.to_jsscene(three, scene)
    mesh = js_scene.getObjectByName(string(objectid(splot)))
    mesh2 = js_scene.getObjectByName(string(objectid(kplot)))

    # init javascript
    evaljs(session,  js"""
        console.log("Hello");
        iter = 1;
        si = 0.0;
        R1 = $(R1);
        R2 = $(R2);
        setInterval(
            function (){
                function randn_bm() {
                    var u = 0, v = 0;
                    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
                    while(v === 0) v = Math.random();
                    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
                }
                var mu = 0.2;
                var mesh = $(mesh);
                var K = $(K);
                var n = $(n);
                var dt = $(dt);
                console.log(iter++);
                var sqrtdt = $(sqrtdt);

                k = iter%K;
                var positions = mesh.geometry.attributes.offset.array;
                var color = mesh.geometry.attributes.color.array;
                console.log(color.length);
                for ( var i = 0; i < n; i++ ) {
                    inew = k*2*n + 2*i;
                    iold = ((K + k - 1)%K)*2*n + 2*i;
                    positions[inew] = (1 - mu*dt)*positions[iold] - 3*dt*positions[iold+1] + si*sqrtdt*randn_bm(); // x
                    positions[inew+1] = (1 - mu*dt)*positions[iold+1] + 3*dt*positions[iold] + si*sqrtdt*randn_bm();
                    color[k*4*n + 4*i] = 1.0;
                    color[k*4*n + 4*i + 1] = 1.0;
                    color[k*4*n + 4*i + 2] = 1.0;
                    color[k*4*n + 4*i + 3] = 1.0;
                    if (Math.random() < 0.01)
                    {
                        positions[inew] = (2*Math.random()-1)*R1;
                        positions[inew+1] = (2*Math.random()-1)*R2;
                    }

                }
                for ( var k = 0; k < K; k++ ) {
                    for ( var i = 0; i < n; i++ ) {
                        color[k*4*n + 4*i + 3] = 0.98*color[k*4*n + 4*i + 3];
                    }
                }
                mesh.geometry.attributes.color.needsUpdate = true;
                mesh.geometry.attributes.offset.needsUpdate = true;

            }
        , 50);
    """)

    onjs(session, sliders.value, js"""function (value){
        si = value;
        var mesh = $(mesh2);
        var positions = mesh.geometry.attributes.offset.array;
        var color = mesh.geometry.attributes.color.array;

        for ( var i = 0, l = positions.length; i < l; i += 2 ) {
                    positions[i+1] = si*Math.sin(positions[i]);
            }
        mesh.geometry.attributes.offset.needsUpdate = true;
        //mesh.geometry.attributes.color.needsUpdate = true;

    }""")

    dom = DOM.div(particlecss, DOM.p(canvas), DOM.p("Parameters"), DOM.div(sliders,  id="slider"),
    DOM.p(nrs))
#    JSServe.onload(session, dom, js"""
#        iter = 1;
#    """)
    println("running...")
    dom
end


app = JSServe.Application(
    dom_handler,
    get(ENV, "WEBIO_SERVER_HOST_URL", "127.0.0.1"),
    parse(Int, get(ENV, "WEBIO_HTTP_PORT", "8081")),
    verbose = false
)
cl() = (close(app), "stopped")
println("Done.")
#
cl()
