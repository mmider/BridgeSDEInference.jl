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
    slider1 = JSServe.Slider(0.01:0.01:1)
    nrs1 = JSServe.NumberInput(0.0)
    linkjs(session, slider1.value, nrs1.value)

    # slider and field for beta
    slider2 = JSServe.Slider(0.01:0.01:10)
    nrs2 = JSServe.NumberInput(0.0)
    linkjs(session, slider2.value, nrs2.value)

    # slider and field for gamma
    slider3 = JSServe.Slider(0.01:0.01:10)
    nrs3 = JSServe.NumberInput(0.0)
    linkjs(session, slider3.value, nrs3.value)

    # slider and field for s
    slider4 = JSServe.Slider(0.01:0.01:10)
    nrs4 = JSServe.NumberInput(0.0)
    linkjs(session, slider4.value, nrs4.value)

    # slider and field for esp
    slider5 = JSServe.Slider(0.01:0.01:10)
    nrs5 = JSServe.NumberInput(0.0)
    linkjs(session, slider5.value, nrs5.value)

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
        console.log("Hello");
        iter = 1;
        eps = 0.1;
        s = -0.8;
        gamma = 1.5;
        beta = 0.0;
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
                    positions[inew] = positions[iold] + dt/eps*((1 - positions[iold]*positions[iold])*positions[iold] - positions[iold+1] - s); // x
                    positions[inew+1] = positions[iold+1] + dt*(-positions[iold+1] + gamma*positions[iold] + beta) + si*sqrtdt*randn_bm();
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

    onjs(session, slider2.value, js"""function (value){
        beta = value;
    }""")

    onjs(session, slider3.value, js"""function (value){
        gamma = value;
    }""")

    onjs(session, slider4.value, js"""function (value){
        s = value;
    }""")

    onjs(session, slider5.value, js"""function (value){
        eps = value;
    }""")
    onjs(session, slider1.value, js"""function (value){
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

    dom = DOM.div(particlecss, DOM.p(canvas), DOM.p("Parameters", DOM.div(slider1,  id="slider1"), DOM.div(slider2,  id="slider2"),
        DOM.div(slider3,  id="slider3"),DOM.div(slider4,  id="slider4"), DOM.div(slider5,  id="slider5")),
        DOM.p(nrs1), DOM.p(nrs2), DOM.p(nrs3), DOM.p(nrs4), DOM.p(nrs5), DOM.p(button))
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
#println("Done.")
#
#cl()
