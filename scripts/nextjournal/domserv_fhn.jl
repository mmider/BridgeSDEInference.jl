using JSServe, WGLMakie, AbstractPlotting
using JSServe: JSServe.DOM, @js_str, onjs
global three, scene

using Colors
using Random
using WGLMakie: scatter, scatter!

# fallback values if statistical analysis is not run
if !@isdefined thetas
   thetas = [0.1, -0.8, 1.5, 0.0, 0.3]
end
if !@isdefined thetalims
   thetalims = [(0.0,1.0), (-5.0,5.0), (0.0,10.0), (0.0,10.0), (0.0,1.0)]
end
WGLMakie.activate!()

using Hyperscript, Markdown
using JSServe, Observables
using JSServe: Application, Session, evaljs, linkjs, div, active_sessions, Asset
using JSServe: @js_str, onjs, Button, TextField, Slider, JSString, Dependency, with_session
using JSServe.DOM

function dom_handler(session, request)
    global three, scene

  	# fetch initial parameter (initial slider settings)
    eps = thetas[1];
    s = thetas[2];
    gamma = thetas[3];
    beta = thetas[4];
    si = thetas[5];
    rebirth = 0.001; # how often a particle is "reborn" at random position
  	sl = 101 # slider sub-divisions

  	# load histogram images to use as slider background
  	sliderbg = [JSServe.Asset(), JSServe.Asset(), JSServe.Asset(), JSServe.Asset(), JSServe.Asset()]

    # slider and field for sigma
    slider5 = JSServe.Slider(range(thetalims[5]..., length=sl), si)
    nrs5 = JSServe.NumberInput(si)
    linkjs(session, slider5.value, nrs5.value)

    # slider and field for beta
    slider4 = JSServe.Slider(range(thetalims[4]..., length=sl), beta)
    nrs4 = JSServe.NumberInput(beta)
    linkjs(session, slider4.value, nrs4.value)

    # slider and field for gamma
    slider3 = JSServe.Slider(range(thetalims[3]..., length=sl), gamma)
    nrs3 = JSServe.NumberInput(gamma)
    linkjs(session, slider3.value, nrs3.value)

    # slider and field for s
    slider2 = JSServe.Slider(range(thetalims[2]..., length=sl), s)
    nrs2 = JSServe.NumberInput(s)
    linkjs(session, slider2.value, nrs2.value)

    # slider and field for eps
    slider1 = JSServe.Slider(range(thetalims[1]..., length=sl), eps)
    nrs1 = JSServe.NumberInput(eps)
    linkjs(session, slider1.value, nrs1.value)

    # slider and field for rebirth
    slider6 = JSServe.Slider(0.0:0.0001:0.005, rebirth)
    nrs6 = JSServe.NumberInput(rebirth)
    linkjs(session, slider6.value, nrs6.value)

    # init
    R = (1.5, 3.0) # plot area
    R1, R2 = R
    limits = FRect(-R[1], -R[2], 2R[1], 2R[2])
    n = 400 # no of particles
    K = 150 # display K past positions of particle fading out
    dt = 0.0005 # time step
    sqrtdt = sqrt(dt)

    particlecss = JSServe.Asset() # style sheet
    ms1 = 0.02 # markersize particles
    ms2 = 0.02 # markersize isokline

  	# plot particles, initially at random positions
    global scene = WGLMakie.scatter(repeat(R1*(2rand(n) .- 1), outer=K), repeat(R2*(2rand(n) .- 1),outer=K), color = fill((:white,0f0), n*K),
        backgroundcolor = RGB{Float32}(0.04, 0.11, 0.22), markersize = ms1,
        glowwidth = 0.005, glowcolor = :white,
        resolution=(500,500), limits = limits,
        )

  	# style plot
    axis = scene[Axis]
    axis[:grid, :linewidth] =  (0.3, 0.3)
    axis[:grid, :linecolor] = (RGBA{Float32}(0.5, 0.7, 1.0, 0.3),RGBA{Float32}(0.5, 0.7, 1.0, 0.3))
    axis[:names][:textsize] = (0.0,0.0)
    axis[:ticks, :textcolor] = (RGBA{Float32}(0.5, 0.7, 1.0, 0.5),RGBA{Float32}(0.5, 0.7, 1.0, 0.5))
    splot = scene[end]

  	# plot isoklines
    WGLMakie.scatter!(scene, -R1:0.01:R1, (-R1:0.01:R1) .- (-R1:0.01:R1).^3 .+ s, color = RGBA{Float32}(0.5, 0.7, 1.0, 0.8), markersize=ms2)
    kplot1 = scene[end]
    WGLMakie.scatter!(scene, -R1:0.01:R1, gamma*(-R1:0.01:R1) .+ beta , color = RGBA{Float32}(0.5, 0.7, 1.0, 0.8), markersize=ms2)
    kplot2 = scene[end]

  	# set up threejs scene
    three, canvas = WGLMakie.three_display(session, scene)
    js_scene = WGLMakie.to_jsscene(three, scene)
    mesh = js_scene.getObjectByName(string(objectid(splot)))
    mesh1 = js_scene.getObjectByName(string(objectid(kplot1)))
    mesh2 = js_scene.getObjectByName(string(objectid(kplot2)))

    # init javascript
    evaljs(session,  js"""
        iter = 1; // iteration number

    		// fetch parameters
        eps = $(eps);
        s = $(s);
        gamma = $(gamma);
        beta = $(beta);
        si = $(si);
        R1 = $(R1);
        R2 = $(R2);
    	  rebirth = $(rebirth);

    		// update functions for isoklines
        updateklinebeta = function (value){
            beta = value;
            var mesh = $(mesh2);
            var positions = mesh.geometry.attributes.offset.array;
            for ( var i = 0, l = positions.length; i < l; i += 2 ) {
                        positions[i+1] = beta + positions[i]*gamma;
                }
            mesh.geometry.attributes.offset.needsUpdate = true;
            //mesh.geometry.attributes.color.needsUpdate = true;
        }
        updateklinegamma = function (value){
            gamma = value;
            var mesh = $(mesh2);
            var positions = mesh.geometry.attributes.offset.array;
            for ( var i = 0, l = positions.length; i < l; i += 2 ) {
                        positions[i+1] = beta + positions[i]*gamma;
                }
            mesh.geometry.attributes.offset.needsUpdate = true;
            //mesh.geometry.attributes.color.needsUpdate = true;
        }
        updateklines = function (value){
            s = value;
            var mesh = $(mesh1);
            var positions = mesh.geometry.attributes.offset.array;
            for ( var i = 0, l = positions.length; i < l; i += 2 ) {
                        positions[i+1] = positions[i] - positions[i]*positions[i]*positions[i] + s;
                }
            mesh.geometry.attributes.offset.needsUpdate = true;
            //mesh.geometry.attributes.color.needsUpdate = true;
        }

    		// move particles every x milliseconds
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
                    positions[inew] = positions[iold] + dt/eps*((1 - positions[iold]*positions[iold])*positions[iold] - positions[iold+1] + s); // x
                    positions[inew+1] = positions[iold+1] + dt*(-positions[iold+1] + gamma*positions[iold] + beta) + si*sqrtdt*randn_bm();
                    color[k*4*n + 4*i] = 1.0;
                    color[k*4*n + 4*i + 1] = 1.0;
                    color[k*4*n + 4*i + 2] = 1.0;
                    color[k*4*n + 4*i + 3] = 1.0;
                    if (Math.random() < rebirth)
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
        , 15); // milliseconds
    """)

  	# react on slider movements

    onjs(session, slider1.value, js"""function (value){
        eps = value;
    }""")
    onjs(session, slider2.value, js"""function (value){
        updateklines(value);
    }""")
      onjs(session, slider3.value, js"""function (value){
        updateklinegamma(value);
    }""")
    onjs(session, slider4.value, js"""function (value){
        updateklinebeta(value);
    }""")
    onjs(session, slider5.value, js"""function (value){
        si = value;
    }""")
  	onjs(session, slider6.value, js"""function (value){
        rebirth = value;
    }""")

  	# set background for sliders
		styles = [Observable("padding-left:0px; padding-top: 10px; height: 50px; width=150px; background-size: 115px 50px; background-repeat: no-repeat;
    background-position: center center; background-image: url($(JSServe.url(sliderbg[j])));") for j in 1:5]

  	# arrange canvas and sliders as html elements
    dom = DOM.div(particlecss, DOM.p(canvas), DOM.p("Parameters"), DOM.table(
    DOM.tr(DOM.td("eps"), DOM.td("s"), DOM.td("gamma"), DOM.td("beta"), DOM.td("sigma")),
    DOM.tr(
        DOM.td(DOM.div(slider1, id="slider1", style=styles[1]), DOM.div(nrs1)),
        DOM.td(DOM.div(slider2, id="slider2", style=styles[2]), DOM.div(nrs2)),
        DOM.td(DOM.div(slider3, id="slider3", style=styles[3]), DOM.div(nrs3)),
        DOM.td(DOM.div(slider4, id="slider4", style=styles[4]), DOM.div(nrs4)),
        DOM.td(DOM.div(slider5, id="slider5", style=styles[5] ), DOM.div(nrs5))
       ),
    DOM.tr(
        DOM.td("rebirth", DOM.div(slider6, id="slider6"), DOM.div(nrs6)),
  	)))
    println("running...")
    dom
end

# attach handler to current session
#JSServe.with_session() do session, request
#    dom_handler(session, request)
#end

app = JSServe.Application(
    dom_handler,
    get(ENV, "WEBIO_SERVER_HOST_URL", "127.0.0.1"),
    parse(Int, get(ENV, "WEBIO_HTTP_PORT", "8081")),
    verbose = false
)

cl() = (close(app), "stopped")
#println("Done.")
#
cl()
