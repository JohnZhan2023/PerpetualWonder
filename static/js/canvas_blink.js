/* ======= Animation config ======= */
// const DATA_DIR   = "./assets/splats/dumpling/";           // where all splats live
let DATA_DIR   = "./assets/splats/dumpling/";           // where all splats live
const STATIC_FN  = "static.splat";
// const T          = 36;                     // number of dynamic frames
let T = 147;
const FRAME_FMT  = f => `${String(f).padStart(3,"0")}_dynamic.splat`;
const PLAY_FPS   = 24; // 10;                      // playback speed
let FRAME_SYNC_ENABLED = false; // true;  // Enable frame synchronization
const ATOMIC_UPDATES = true;   // Enable atomic updates to avoid blinking
/* ================================= */

let defaultViewMatrix = [
    0.32006332420023925, 0.16494937955842429, -0.932926133563201, 0,
    0.947396151829789, -0.05572562928851899, 0.3151748494701092, 0,
    -1.3877787807814457e-17, -0.9847265389049336, -0.174108137593596, 0,
    -0.10401135625176541, 0.5276397360604658, 2.759314731410553, 1
];
// let defaultViewMatrix = [1,0,0,0,
//                             0,1,0,0,
//                             0,0,1,0,
//                             0,0,0,1];
let yaw = 0;   // Rotation around the Y-axis
let pitch = 0; // Rotation around the X-axis
let movement = [0, 0, 0]; // Movement vector initialized to 0,0,0
const render_width = 1280; // 512;
const render_height = 704; //512;
let bbox = [-9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999];
let radius = 9999;
const rowLength = 32;
const ROW_LEN = 32;    

const cameras = [
    {
        id: 0,
        position: [
            -0.5, 1.0, 0   // +left, +up, +forward
        ],
        rotation: [
            [-1, 0, 0],
            [0., 1, 0],
            [0, 0, -1],
        ],
        fy: 960,
        fx: 960,
    }
];

const sceneConfigs = {
    dumpling: {
        DATA_DIR: "./assets/splats/dumpling/",
        T: 147,
        movement: [0, 0, 0],
        yaw: 0,  // 90度旋转
        pitch: 0,
        bbox: [
            -0.8, 0.8,
            -0.1, 0.5,
            -0.05, 0.1,
            -0.25, 0.25
        ],
        fx: 1491.38,
        fy: 1506.07,
        cx: 640,
        cy: 352,
    }
};

function getProjectionMatrix(fx, fy, width, height, cx = null, cy = null) {
    const znear = 0.01;
    const zfar = 100;
    
    // 如果没有提供 cx/cy，使用图像中心作为默认值
    if (cx === null) cx = width / 2;
    if (cy === null) cy = height / 2;
    
    return [
        [(2 * fx) / width, 0, 0, 0],
        [0, -(2 * fy) / height, 0, 0],
        [1 - (2 * cx) / width, -1 + (2 * cy) / height, zfar / (zfar - znear), 1],
        [0, 0, -(zfar * znear) / (zfar - znear), 0],
    ].flat();
}

function getViewMatrix(camera) {
    const R = camera.rotation.flat();
    const t = camera.position;
    const camToWorld = [
        [R[0], R[1], R[2], 0],
        [R[3], R[4], R[5], 0],
        [R[6], R[7], R[8], 0],
        [
            -t[0] * R[0] - t[1] * R[3] - t[2] * R[6],
            -t[0] * R[1] - t[1] * R[4] - t[2] * R[7],
            -t[0] * R[2] - t[1] * R[5] - t[2] * R[8],
            1,
        ],
    ].flat();
    return camToWorld;
}

function multiply4(a, b) {
    return [
        b[0] * a[0] + b[1] * a[4] + b[2] * a[8] + b[3] * a[12],
        b[0] * a[1] + b[1] * a[5] + b[2] * a[9] + b[3] * a[13],
        b[0] * a[2] + b[1] * a[6] + b[2] * a[10] + b[3] * a[14],
        b[0] * a[3] + b[1] * a[7] + b[2] * a[11] + b[3] * a[15],
        b[4] * a[0] + b[5] * a[4] + b[6] * a[8] + b[7] * a[12],
        b[4] * a[1] + b[5] * a[5] + b[6] * a[9] + b[7] * a[13],
        b[4] * a[2] + b[5] * a[6] + b[6] * a[10] + b[7] * a[14],
        b[4] * a[3] + b[5] * a[7] + b[6] * a[11] + b[7] * a[15],
        b[8] * a[0] + b[9] * a[4] + b[10] * a[8] + b[11] * a[12],
        b[8] * a[1] + b[9] * a[5] + b[10] * a[9] + b[11] * a[13],
        b[8] * a[2] + b[9] * a[6] + b[10] * a[10] + b[11] * a[14],
        b[8] * a[3] + b[9] * a[7] + b[10] * a[11] + b[11] * a[15],
        b[12] * a[0] + b[13] * a[4] + b[14] * a[8] + b[15] * a[12],
        b[12] * a[1] + b[13] * a[5] + b[14] * a[9] + b[15] * a[13],
        b[12] * a[2] + b[13] * a[6] + b[14] * a[10] + b[15] * a[14],
        b[12] * a[3] + b[13] * a[7] + b[14] * a[11] + b[15] * a[15],
    ];
}

function invert4(a) {
    let b00 = a[0] * a[5] - a[1] * a[4];
    let b01 = a[0] * a[6] - a[2] * a[4];
    let b02 = a[0] * a[7] - a[3] * a[4];
    let b03 = a[1] * a[6] - a[2] * a[5];
    let b04 = a[1] * a[7] - a[3] * a[5];
    let b05 = a[2] * a[7] - a[3] * a[6];
    let b06 = a[8] * a[13] - a[9] * a[12];
    let b07 = a[8] * a[14] - a[10] * a[12];
    let b08 = a[8] * a[15] - a[11] * a[12];
    let b09 = a[9] * a[14] - a[10] * a[13];
    let b10 = a[9] * a[15] - a[11] * a[13];
    let b11 = a[10] * a[15] - a[11] * a[14];
    let det =
        b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) return null;
    return [
        (a[5] * b11 - a[6] * b10 + a[7] * b09) / det,
        (a[2] * b10 - a[1] * b11 - a[3] * b09) / det,
        (a[13] * b05 - a[14] * b04 + a[15] * b03) / det,
        (a[10] * b04 - a[9] * b05 - a[11] * b03) / det,
        (a[6] * b08 - a[4] * b11 - a[7] * b07) / det,
        (a[0] * b11 - a[2] * b08 + a[3] * b07) / det,
        (a[14] * b02 - a[12] * b05 - a[15] * b01) / det,
        (a[8] * b05 - a[10] * b02 + a[11] * b01) / det,
        (a[4] * b10 - a[5] * b08 + a[7] * b06) / det,
        (a[1] * b08 - a[0] * b10 - a[3] * b06) / det,
        (a[12] * b04 - a[13] * b02 + a[15] * b00) / det,
        (a[9] * b02 - a[8] * b04 - a[11] * b00) / det,
        (a[5] * b07 - a[4] * b09 - a[6] * b06) / det,
        (a[0] * b09 - a[1] * b07 + a[2] * b06) / det,
        (a[13] * b01 - a[12] * b03 - a[14] * b00) / det,
        (a[8] * b03 - a[9] * b01 + a[10] * b00) / det,
    ];
}

function rotate4(a, rad, x, y, z) {
    let len = Math.hypot(x, y, z);
    x /= len;
    y /= len;
    z /= len;
    let s = Math.sin(rad);
    let c = Math.cos(rad);
    let t = 1 - c;
    let b00 = x * x * t + c;
    let b01 = y * x * t + z * s;
    let b02 = z * x * t - y * s;
    let b10 = x * y * t - z * s;
    let b11 = y * y * t + c;
    let b12 = z * y * t + x * s;
    let b20 = x * z * t + y * s;
    let b21 = y * z * t - x * s;
    let b22 = z * z * t + c;
    return [
        a[0] * b00 + a[4] * b01 + a[8] * b02,
        a[1] * b00 + a[5] * b01 + a[9] * b02,
        a[2] * b00 + a[6] * b01 + a[10] * b02,
        a[3] * b00 + a[7] * b01 + a[11] * b02,
        a[0] * b10 + a[4] * b11 + a[8] * b12,
        a[1] * b10 + a[5] * b11 + a[9] * b12,
        a[2] * b10 + a[6] * b11 + a[10] * b12,
        a[3] * b10 + a[7] * b11 + a[11] * b12,
        a[0] * b20 + a[4] * b21 + a[8] * b22,
        a[1] * b20 + a[5] * b21 + a[9] * b22,
        a[2] * b20 + a[6] * b21 + a[10] * b22,
        a[3] * b20 + a[7] * b21 + a[11] * b22,
        ...a.slice(12, 16),
    ];
}

function translate4(a, x, y, z) {
    return [
        ...a.slice(0, 12),
        a[0] * x + a[4] * y + a[8] * z + a[12],
        a[1] * x + a[5] * y + a[9] * z + a[13],
        a[2] * x + a[6] * y + a[10] * z + a[14],
        a[3] * x + a[7] * y + a[11] * z + a[15],
    ];
}

function createWorker(self) {
    let buffer;
    let vertexCount = 0;
    let viewProj;
    let currentFrameId = -1; // Track which frame we're processing
    const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
    let lastProj = [];
    let depthIndex = new Uint32Array();
    let lastVertexCount = 0;
    let sortRunning = false;  // Initialize here and don't declare again below

    var _floatView = new Float32Array(1);
    var _int32View = new Int32Array(_floatView.buffer);

    function floatToHalf(float) {
        _floatView[0] = float;
        var f = _int32View[0];

        var sign = (f >> 31) & 0x0001;
        var exp = (f >> 23) & 0x00ff;
        var frac = f & 0x007fffff;

        var newExp;
        if (exp == 0) {
            newExp = 0;
        } else if (exp < 113) {
            newExp = 0;
            frac |= 0x00800000;
            frac = frac >> (113 - exp);
            if (frac & 0x01000000) {
                newExp = 1;
                frac = 0;
            }
        } else if (exp < 142) {
            newExp = exp - 112;
        } else {
            newExp = 31;
            frac = 0;
        }

        return (sign << 15) | (newExp << 10) | (frac >> 13);
    }

    function packHalf2x16(x, y) {
        return (floatToHalf(x) | (floatToHalf(y) << 16)) >>> 0;
    }

    let textureGenerated = false;
    let sortGenerated = false;
    let pendingTexdata = null;
    let pendingDepthIndex = null;
    
    function checkFrameReady() {
        if (textureGenerated && sortGenerated && pendingTexdata && pendingDepthIndex) {
            // console.log(`Worker: Frame ${currentFrameId} ready, sending to main thread`);
            self.postMessage({ 
                type: 'frame_ready',
                frameId: currentFrameId,
                texdata: pendingTexdata,
                texwidth: 1024 * 2, // Match what's in generateTexture
                texheight: Math.ceil((2 * vertexCount) / (1024 * 2)),
                depthIndex: pendingDepthIndex,
                vertexCount: vertexCount
            }, [pendingTexdata.buffer, pendingDepthIndex.buffer]);
            
            textureGenerated = false;
            sortGenerated = false;
            pendingTexdata = null;
            pendingDepthIndex = null;
        }
    }

    function generateTexture() {
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);
        const u_buffer = new Uint8Array(buffer);

        var texwidth = 1024 * 2; 
        var texheight = Math.ceil((2 * vertexCount) / texwidth);
        var texdata = new Uint32Array(texwidth * texheight * 4);
        texdata.fill(0);
        var texdata_c = new Uint8Array(texdata.buffer);
        var texdata_f = new Float32Array(texdata.buffer);

        // Here we convert from a .splat file buffer into a texture
        // With a little bit more foresight perhaps this texture file
        // should have been the native format as it'd be very easy to
        // load it into webgl.
        
        for (let i = 0; i < vertexCount; i++) {
            // x, y, z
            texdata_f[8 * i + 0] = f_buffer[8 * i + 0];
            texdata_f[8 * i + 1] = f_buffer[8 * i + 1];
            texdata_f[8 * i + 2] = f_buffer[8 * i + 2];

            // r, g, b, a
            texdata_c[4 * (8 * i + 7) + 0] = u_buffer[32 * i + 24 + 0];
            texdata_c[4 * (8 * i + 7) + 1] = u_buffer[32 * i + 24 + 1];
            texdata_c[4 * (8 * i + 7) + 2] = u_buffer[32 * i + 24 + 2];
            texdata_c[4 * (8 * i + 7) + 3] = u_buffer[32 * i + 24 + 3];

            // quaternions
            let scale = [
                f_buffer[8 * i + 3 + 0],
                f_buffer[8 * i + 3 + 1],
                f_buffer[8 * i + 3 + 2],
            ];
            let rot = [
                (u_buffer[32 * i + 28 + 0] - 128) / 128,
                (u_buffer[32 * i + 28 + 1] - 128) / 128,
                (u_buffer[32 * i + 28 + 2] - 128) / 128,
                (u_buffer[32 * i + 28 + 3] - 128) / 128,
            ];

            // Compute the matrix product of S and R (M = S * R)
            const M = [
                1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3]),
                2.0 * (rot[1] * rot[2] + rot[0] * rot[3]),
                2.0 * (rot[1] * rot[3] - rot[0] * rot[2]),

                2.0 * (rot[1] * rot[2] - rot[0] * rot[3]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[3] * rot[3]),
                2.0 * (rot[2] * rot[3] + rot[0] * rot[1]),

                2.0 * (rot[1] * rot[3] + rot[0] * rot[2]),
                2.0 * (rot[2] * rot[3] - rot[0] * rot[1]),
                1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2]),
            ].map((k, i) => k * scale[Math.floor(i / 3)]);

            const sigma = [
                M[0] * M[0] + M[3] * M[3] + M[6] * M[6],
                M[0] * M[1] + M[3] * M[4] + M[6] * M[7],
                M[0] * M[2] + M[3] * M[5] + M[6] * M[8],
                M[1] * M[1] + M[4] * M[4] + M[7] * M[7],
                M[1] * M[2] + M[4] * M[5] + M[7] * M[8],
                M[2] * M[2] + M[5] * M[5] + M[8] * M[8],
            ];

            texdata[8 * i + 4] = packHalf2x16(4 * sigma[0], 4 * sigma[1]);
            texdata[8 * i + 5] = packHalf2x16(4 * sigma[2], 4 * sigma[3]);
            texdata[8 * i + 6] = packHalf2x16(4 * sigma[4], 4 * sigma[5]);

            // if (i == 1) {
            //     console.log("Color bytes:", u_buffer.slice(32 * i + 24, 32 * i + 28));
            // }
        }

        pendingTexdata = texdata;
        textureGenerated = true;
        checkFrameReady();
    }

    function runSort(viewProj) {
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);
        
        // Always generate texture when vertex count changes
        if (lastVertexCount != vertexCount) {
            generateTexture();
        }
        lastVertexCount = vertexCount;

        // Always run the sort
        // console.time(`sort-frame-${currentFrameId}`);
        let maxDepth = -Infinity;
        let minDepth = Infinity;
        let sizeList = new Int32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++) {
            let depth =
                ((viewProj[2] * f_buffer[8 * i + 0] +
                    viewProj[6] * f_buffer[8 * i + 1] +
                    viewProj[10] * f_buffer[8 * i + 2]) *
                    4096) |
                0;
            sizeList[i] = depth;
            if (depth > maxDepth) maxDepth = depth;
            if (depth < minDepth) minDepth = depth;
        }
        
        // This is a 16 bit single-pass counting sort
        let depthInv = (256 * 256) / (maxDepth - minDepth);
        let counts0 = new Uint32Array(256 * 256);

        for (let i = 0; i < vertexCount; i++) {
            sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
            counts0[sizeList[i]]++;
        }
        let starts0 = new Uint32Array(256 * 256);
        for (let i = 1; i < 256 * 256; i++)
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        depthIndex = new Uint32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++)
            depthIndex[starts0[sizeList[i]]++] = i;

        // console.timeEnd(`sort-frame-${currentFrameId}`);

        lastProj = viewProj;
        
        pendingDepthIndex = depthIndex;
        sortGenerated = true;
        checkFrameReady();
    }

    const throttledSort = () => {
        if (!sortRunning) {
            sortRunning = true;
            let lastView = viewProj;
            runSort(lastView);
            setTimeout(() => {
                sortRunning = false;
                if (lastView !== viewProj) {
                    throttledSort();
                }
            }, 0);
        }
    };

    self.onmessage = (e) => {
        if (e.data.buffer) {
            buffer = e.data.buffer;
            vertexCount = e.data.vertexCount;
            currentFrameId = e.data.frameId || 0;
            
            // Reset the ready flags
            textureGenerated = false;
            sortGenerated = false;
            
            // Generate texture and run sort if possible
            generateTexture();
            if (viewProj) {
                runSort(viewProj);
            }
        } else if (e.data.view) {
            viewProj = e.data.view;
            
            if (buffer && !sortGenerated) {
                runSort(viewProj);
            }
        } else if (e.data.clear) {
            buffer = null;
            vertexCount = 0;
            lastVertexCount = 0;
            depthIndex = new Uint32Array();
            lastProj = [];
            pendingTexdata = null;
            pendingDepthIndex = null;
            textureGenerated = false;
            sortGenerated = false;
            return;
        }
    };
}

const vertexShaderSource = `
#version 300 es
precision highp float;
precision highp int;

uniform highp usampler2D u_texture;
uniform mat4 projection, view;
uniform vec2 focal;
uniform vec2 viewport;

in vec2 position;
in int index;

out vec4 vColor;
out vec2 vPosition;

void main () {
uvec4 cen = texelFetch(u_texture, ivec2((uint(index) & 0x3ffu) << 1, uint(index) >> 10), 0);
vec4 cam = view * vec4(uintBitsToFloat(cen.xyz), 1);
vec4 pos2d = projection * cam;

float clip = 1.2 * pos2d.w;
if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
return;
}

uvec4 cov = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 1) | 1u, uint(index) >> 10), 0);
vec2 u1 = unpackHalf2x16(cov.x), u2 = unpackHalf2x16(cov.y), u3 = unpackHalf2x16(cov.z);
mat3 Vrk = mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);

mat3 J = mat3(
focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z), 
0., -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z), 
0., 0., 0.
);

mat3 T = transpose(mat3(view)) * J;
mat3 diag = mat3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
float dilation = 0.0;
mat3 cov2d = transpose(T) * Vrk * T + diag * dilation;

float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
float lambda1 = mid + radius, lambda2 = mid - radius;

if(lambda2 < 0.0) return;
vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

vColor = clamp(pos2d.z/pos2d.w+1.0, 0.0, 1.0) * vec4((cov.w) & 0xffu, (cov.w >> 8) & 0xffu, (cov.w >> 16) & 0xffu, (cov.w >> 24) & 0xffu) / 255.0;
vPosition = position;

vec2 vCenter = vec2(pos2d) / pos2d.w;
gl_Position = vec4(
vCenter 
+ position.x * majorAxis / viewport 
+ position.y * minorAxis / viewport, 0.0, 1.0);

}
`.trim();

const fragmentShaderSource = `
#version 300 es
precision highp float;

in vec4 vColor;
in vec2 vPosition;

out vec4 fragColor;

void main () {
float A = -dot(vPosition, vPosition);
if (A < -4.0) discard;
float B = exp(A) * vColor.a;
fragColor = vec4(B * vColor.rgb, B);
}

`.trim();

let viewMatrix = defaultViewMatrix;

async function fetchUint8(url) {
    const r = await fetch(url, {mode:"cors", cache:"no-cache"});
    if (!r.ok) throw Error(`${r.status} ${url}`);
    return new Uint8Array(await r.arrayBuffer());
}

// Track the current animation frame request so we can cancel it
let currentAnimationFrame = null;

// Function to completely clear the scene and stop all rendering
function clearScene() {
    // Cancel any running animation frames
    if (currentAnimationFrame) {
        cancelAnimationFrame(currentAnimationFrame);
        currentAnimationFrame = null;
    }
    
    // Clear worker state
    if (window.worker) {
        // Send an empty buffer with zero vertices
        const emptyBuffer = new Uint8Array(0);
        window.worker.postMessage({
            buffer: emptyBuffer.buffer,
            vertexCount: 0
        });
        
        // Explicitly clear the worker
        window.worker.postMessage({ clear: true });
        
        // Terminate the worker completely
        window.worker.terminate();
        window.worker = null;
    }
    
    // Reset global state variables
    window.vertexCount = 0;
    window.currentFrameProcessing = false;
    window.frameIdx = 0;
}

// Modified applySceneConfig to properly clear the scene
function applySceneConfig(sceneName) {
    if (!sceneConfigs[sceneName]) {
        console.error(`Scene config "${sceneName}" not found`);
        return false;
    }
    
    // Update UI elements first
    const message = document.getElementById("message");
    const spinner = document.getElementById("spinner");
    if (message) {
        message.style.display = "block";
        message.innerText = `Clearing previous scene...`;
    }
    if (spinner) spinner.style.display = "block";
    
    // First, completely clear the current scene
    clearScene();
    
    // Then update configuration with a delay to ensure cleanup is complete
    setTimeout(() => {
        const config = sceneConfigs[sceneName];
        
        // Update global variables with scene config
        DATA_DIR = config.DATA_DIR;
        T = config.T;
        movement = [...config.movement];
        yaw = config.yaw;
        pitch = config.pitch;
        bbox = [...config.bbox];
        const fxValue = config.fx;
        radius = config.radius;
        
        // Store the selected fx for main() to use
        window.selectedFx = fxValue;
        
        if (message) {
            message.innerText = `Loading ${sceneName} scene...`;
        }
        
        // Now load the new scene
        main().catch(err => {
            console.error(`Error loading ${sceneName} scene:`, err);
            if (message) message.innerText = err.toString();
            if (spinner) spinner.style.display = "none";
        });
    }, 200); // Use a longer delay to ensure cleanup completes
    
    return true;
}

async function main() {
    /* ---------------- DOM helpers ---------------- */
    const canvas      = document.getElementById("canvas");
    const messageEl   = document.getElementById("message");
    const spinnerEl   = document.getElementById("spinner");
    
    /* ---------------- download splats ---------------- */
    if (spinnerEl) spinnerEl.style.display = "block";
    if (messageEl) messageEl.innerText = "Downloading splats…";
    
    // Start with fresh data downloads
    const staticBytes = await fetchUint8(DATA_DIR + STATIC_FN);
    
    // Show progress for dynamic frames
    if (messageEl) messageEl.innerText = "Downloading dynamic frames...";
    const dynamicRaw = [];
    
    // Load frames in batches to avoid memory pressure
    const BATCH_SIZE = 5;
    for (let batch = 0; batch < Math.ceil(T/BATCH_SIZE); batch++) {
        const start = batch * BATCH_SIZE;
        const end = Math.min(start + BATCH_SIZE, T);
        
        // Process a batch
        const batchPromises = [];
        for (let i = start; i < end; i++) {
            batchPromises.push(fetchUint8(DATA_DIR + FRAME_FMT(i)));
        }
        
        // Wait for batch to complete
        const batchResults = await Promise.all(batchPromises);
        for (const result of batchResults) {
            dynamicRaw.push(result);
        }
        
        if (messageEl) messageEl.innerText = `Loaded ${dynamicRaw.length}/${T} frames...`;
    }

    if (spinnerEl) spinnerEl.style.display = "none";
    if (messageEl) messageEl.innerText = "";

    /* ---------------- WebGL boilerplate ---------------- */
    const gl = canvas.getContext("webgl2", { antialias: false });

    const vShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vShader, vertexShaderSource);
    gl.compileShader(vShader);

    const fShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fShader, fragmentShaderSource);
    gl.compileShader(fShader);

    const program = gl.createProgram();
    gl.attachShader(program, vShader);
    gl.attachShader(program, fShader);
    gl.linkProgram(program);
    gl.useProgram(program);

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);

    /* uniforms */
    const u_projection = gl.getUniformLocation(program, "projection");
    const u_view       = gl.getUniformLocation(program, "view");
    const u_focal      = gl.getUniformLocation(program, "focal");
    const u_viewport   = gl.getUniformLocation(program, "viewport");

    /* full‑screen quad */
    const quad = new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]);
    const quadBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
    const a_position = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(a_position);
    gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);

    /* index buffer (instanced) */
    const indexBuf = gl.createBuffer();
    const a_index  = gl.getAttribLocation(program, "index");
    gl.enableVertexAttribArray(a_index);
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
    gl.vertexAttribIPointer(a_index, 1, gl.INT, false, 0, 0);
    gl.vertexAttribDivisor(a_index, 1);

    /* texture placeholder */
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(gl.getUniformLocation(program, "u_texture"), 0);

    /* viewport */
    const resize = () => {
        canvas.width  = render_width;
        canvas.height = render_height;
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.uniform2fv(u_viewport, new Float32Array([render_width, render_height]));
    };
    window.addEventListener("resize", resize);
    resize();

    /* ---------------- camera ---------------- */
    let active_camera = JSON.parse(JSON.stringify(cameras[0]));
    
    // Use the fx value from selected scene config if available
    if (window.selectedFx) {
        active_camera.fx = active_camera.fy = window.selectedFx;
    }
    
    const use_intrinsics = (cam) => {
        gl.uniform2fv(u_focal, new Float32Array([cam.fx, cam.fy]));
        const proj = getProjectionMatrix(cam.fx, cam.fy, render_width, render_height, cam.cx, cam.cy);
        gl.uniformMatrix4fv(u_projection, false, proj);
        projectionMatrix = proj;  // store for worker viewProj later
    };
    const use_extrinsics = (cam) => {
        viewMatrix = getViewMatrix(cam);
        gl.uniformMatrix4fv(u_view, false, viewMatrix);
    };
    let projectionMatrix;
    viewMatrix = defaultViewMatrix;

    use_intrinsics(active_camera);
    use_extrinsics(active_camera);

    // 应用绕法线方向（Z轴）旋转90度
    let inv = invert4(viewMatrix);
    inv = rotate4(inv, Math.PI / 2, 0, 0, 1); // 绕Z轴旋转90度（法线方向）
    viewMatrix = invert4(inv);
    gl.uniformMatrix4fv(u_view, false, viewMatrix);

    /* ---------------- worker setup ---------------- */
    // Create a modified worker that better handles memory
    const workerCode = createWorker.toString().replace(
        'self.onmessage = (e) => {',
        `self.onmessage = (e) => {
            // Reset state when receiving a clear command
            if (e.data.clear) {
                buffer = null;
                vertexCount = 0;
                lastVertexCount = 0;
                depthIndex = new Uint32Array();
                lastProj = [];
                pendingTexdata = null;
                pendingDepthIndex = null;
                textureGenerated = false;
                sortGenerated = false;
                return;
            }
        `
    );
    
    const worker = new Worker(URL.createObjectURL(new Blob([`(${workerCode})(self)`], { type: "application/javascript" })));
    window.worker = worker; // Make it globally accessible for cleanup
    
    let vertexCount = 0;
    let currentFrameProcessing = false;
    let lastFrameProcessed = -1;
    
    // Improved resetWorker function to clear data more completely
    function resetWorker() {
        // Clear the worker state
        worker.postMessage({ clear: true });
        
        // Clear vertex data
        const emptyBuffer = new Uint8Array(0);
        worker.postMessage({
            buffer: emptyBuffer.buffer,
            vertexCount: 0,
        });
        
        // Reset vertex count to avoid rendering artifacts
        vertexCount = 0;
        
        // Reset frame processing state
        currentFrameProcessing = false;
        
        // Clear WebGL texture and index buffer to prevent showing old data
        if (gl) {
            // Clear the texture by uploading a 1x1 empty texture
            const emptyTexture = new Uint32Array(4);
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, 1, 1, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, emptyTexture);
            
            // Clear the index buffer
            gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
            gl.bufferData(gl.ARRAY_BUFFER, new Uint32Array(0), gl.DYNAMIC_DRAW);
            
            // Clear the screen
            gl.clear(gl.COLOR_BUFFER_BIT);
        }
    }
    
    // Handler for worker messages with improved error recovery
    worker.onmessage = (e) => {
        try {
            if (e.data.type === 'frame_ready') {
                // Update texture
                gl.bindTexture(gl.TEXTURE_2D, texture);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, e.data.texwidth, e.data.texheight, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, e.data.texdata);
                
                // Update index buffer
                gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
                gl.bufferData(gl.ARRAY_BUFFER, e.data.depthIndex, gl.DYNAMIC_DRAW);
                vertexCount = e.data.vertexCount;
                
                lastFrameProcessed = e.data.frameId;
                currentFrameProcessing = false;
                
                if (messageEl && e.data.frameId === 0) {
                    messageEl.innerText = "First frame rendered";
                    setTimeout(() => { 
                        if (messageEl) messageEl.innerText = "";
                        if (spinnerEl) spinnerEl.style.display = "none";
                        
                        // Update UI to show canvas is ready
                        const instructions = document.getElementById("canvas-instructions");
                        const buttonText = document.getElementById("canvas-button-text");
                        if (instructions) instructions.innerText = "Interactive Viewer (Use WASD to move, IJKL to look around)";
                        if (buttonText) buttonText.innerText = "Interactive Viewer Loaded";
                    }, 1000);
                }
            } else if (e.data.depthIndex) {
                // Process individual depth index update
                gl.bindBuffer(gl.ARRAY_BUFFER, indexBuf);
                gl.bufferData(gl.ARRAY_BUFFER, e.data.depthIndex, gl.DYNAMIC_DRAW);
                vertexCount = e.data.vertexCount;
                
                if (!ATOMIC_UPDATES) {
                    currentFrameProcessing = false;
                }
                
                console.log("Received depthIndex update");
            } else if (e.data.texdata) {
                // Process individual texture update
                gl.bindTexture(gl.TEXTURE_2D, texture);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
                gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, e.data.texwidth, e.data.texheight, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, e.data.texdata);
                
                console.log("Received texture update");
            }
        } catch (err) {
            console.error("Error in worker message handler:", err);
            currentFrameProcessing = false;
            resetWorker();
        }
    };

    const keys = new Set();
    window.addEventListener("keydown", (e) => { if (document.activeElement === document.body) keys.add(e.code); });
    window.addEventListener("keyup",   (e) => keys.delete(e.code));

    // Add touch controls
    let touchStartY = 0;
    let touchStartX = 0;
    let isTwoFingerTouch = false;
    const TOUCH_SENSITIVITY_THRESHOLD = 10; // pixels

    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        touchStartY = e.touches[0].clientY;
        touchStartX = e.touches[0].clientX;
        isTwoFingerTouch = e.touches.length === 2;
    });

    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        const touchEndY = e.touches[0].clientY;
        const touchEndX = e.touches[0].clientX;
        const deltaY = touchEndY - touchStartY;
        const deltaX = touchEndX - touchStartX;
        
        if (Math.abs(deltaY) > TOUCH_SENSITIVITY_THRESHOLD || Math.abs(deltaX) > TOUCH_SENSITIVITY_THRESHOLD) {
            if (isTwoFingerTouch) {
                // Two-finger touch for pitch and yaw
                if (Math.abs(deltaY) > Math.abs(deltaX)) {
                    // Vertical movement
                    keys.delete("KeyI");
                    keys.delete("KeyK");
                    if (deltaY < 0) keys.add("KeyI");
                    else keys.add("KeyK");
                } else {
                    // Horizontal movement
                    keys.delete("KeyJ");
                    keys.delete("KeyL");
                    if (deltaX < 0) keys.add("KeyJ");
                    else keys.add("KeyL");
                }
            } else {
                // Single-finger touch for movement
                if (Math.abs(deltaY) > Math.abs(deltaX)) {
                    // Vertical movement
                    keys.delete("KeyW");
                    keys.delete("KeyS");
                    if (deltaY < 0) keys.add("KeyW");
                    else keys.add("KeyS");
                } else {
                    // Horizontal movement
                    keys.delete("KeyA");
                    keys.delete("KeyD");
                    if (deltaX < 0) keys.add("KeyA");
                    else keys.add("KeyD");
                }
            }
            
            touchStartY = touchEndY;
            touchStartX = touchEndX;
        }
    });

    canvas.addEventListener('touchend', (e) => {
        e.preventDefault();
        // Remove all touch-related keys when touch ends
        keys.delete("KeyW");
        keys.delete("KeyS");
        keys.delete("KeyA");
        keys.delete("KeyD");
        keys.delete("KeyI");
        keys.delete("KeyK");
        keys.delete("KeyJ");
        keys.delete("KeyL");
        isTwoFingerTouch = false;
    });

    let frameIdx = 0;
    const frameMs = 1000 / PLAY_FPS;
    let lastSwap = performance.now();

    // Diagnostic values to help track memory issues
    let frameId = 0;
    let lastMemoryCheck = performance.now();
    let consecutiveErrors = 0;
    
    // Calculate FPS for debugging
    let frameCount = 0;
    let lastFpsUpdate = 0;

    // Modified loop with improved memory management
    const loop = (now) => {
        currentAnimationFrame = requestAnimationFrame(loop);
        
        // Calculate and display actual FPS
        frameCount++;
        if (now - lastFpsUpdate >= 1000) {
            const fps = Math.round((frameCount * 1000) / (now - lastFpsUpdate));
            frameCount = 0;
            lastFpsUpdate = now;
        }
        
        // First handle movement and view updates
        let inv = invert4(defaultViewMatrix);
        
        if (keys.has("KeyJ")) yaw -= 0.01;
        if (keys.has("KeyL")) yaw += 0.01;
        if (keys.has("KeyI")) pitch += 0.005;
        if (keys.has("KeyK")) pitch -= 0.005;

        if (keys.has("KeyQ")) {
            const msg = document.getElementById("message");
            if (msg) {
                msg.innerText = 'movement:' + movement + ' yaw:' + yaw + ' pitch:' + pitch + ' focal:' + active_camera.fx;
            }
            console.log("[Camera Debug] movement:", movement, "yaw:", yaw, "pitch:", pitch, "focal:", active_camera.fx);
            console.log("[Camera Debug] rotation matrix (view):", viewMatrix);
        }

        pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, pitch));
        pitch = Math.max(bbox[4], Math.min(bbox[5], pitch));
        yaw = Math.max(bbox[6], Math.min(bbox[7], yaw));

        // Compute movement vector increment based on yaw
        movement_speed = 0.7;
        let dx = 0, dz = 0, dy = 0;
        if (keys.has("KeyW")) dz += 0.01 * movement_speed;
        if (keys.has("KeyS")) dz -= 0.01 * movement_speed;
        if (keys.has("KeyA")) dx -= 0.01 * movement_speed;
        if (keys.has("KeyD")) dx += 0.01 * movement_speed;

        // Convert dx and dz into world coordinates based on yaw
        let forward = [Math.sin(yaw) * dz, 0, Math.cos(yaw) * dz];
        let right = [Math.sin(yaw + Math.PI / 2) * dx, 0, Math.cos(yaw + Math.PI / 2) * dx];

        // Update movement vector
        movement[0] += forward[0] + right[0];
        movement[0] = Math.max(bbox[0], Math.min(bbox[1], movement[0]));
        movement[1] += forward[1] + right[1] + dy; // This should generally remain 0 in a FPS
        movement[2] += forward[2] + right[2];
        movement[2] = Math.max(bbox[2], Math.min(bbox[3], movement[2]));

        // If the distance of movement if higher than radius, then normalize it
        let distance = Math.hypot(movement[0], movement[1], movement[2]);
        if (distance > radius) {
            movement = movement.map((k) => k / distance * radius);
        }

        // Apply translation based on movement vector
        inv = translate4(inv, ...movement);
        
        // Apply rotations
        inv = rotate4(inv, yaw, 0, 1, 0); // Yaw around the Y-axis
        inv = rotate4(inv, pitch, 1, 0, 0); // Pitch around the X-axis
        // Compute the view matrix
        viewMatrix = invert4(inv);
        let inv2 = invert4(viewMatrix);
        let actualViewMatrix = invert4(inv2);
        const viewProj = multiply4(projectionMatrix, actualViewMatrix);
        gl.uniformMatrix4fv(u_view, false, viewMatrix);
        
        // Handle frame changes with better memory management
        if (now - lastSwap > frameMs && !currentFrameProcessing) {
            try {
                lastSwap = now;
                frameIdx = (frameIdx + 1) % T;
                frameId++;
                
                // Memory safety check - if we're experiencing issues, force a reset
                if (now - lastMemoryCheck > 30000) { // Every 30 seconds
                    resetWorker();
                    lastMemoryCheck = now;
                    consecutiveErrors = 0;
                }
                
                // Create a fresh buffer for this frame only when needed
                currentFrameProcessing = true;
                const dyn = dynamicRaw[frameIdx];
                
                if (!dyn) {
                    console.error(`Missing dynamic frame ${frameIdx}`);
                    currentFrameProcessing = false;
                    return;
                }
                
                // Create a new buffer for this frame that will be transferred
                const frameBuffer = new ArrayBuffer(staticBytes.length + dyn.length);
                const frameData = new Uint8Array(frameBuffer);
                
                // Copy data into the buffer
                frameData.set(staticBytes);
                frameData.set(dyn, staticBytes.length);
                
                // Send buffer with current frameId - transfer ownership to reduce memory usage
                worker.postMessage({ 
                    buffer: frameBuffer, 
                    vertexCount: frameData.length / ROW_LEN,
                    frameId: frameId
                }, [frameBuffer]); // Transfer buffer ownership
                
                // Also send the current view for sorting
                worker.postMessage({ 
                    view: viewProj, 
                    frameId: frameId
                });
            } catch (err) {
                console.error("Error preparing frame:", err);
                currentFrameProcessing = false;
                consecutiveErrors++;
                
                if (consecutiveErrors > 3) {
                    // If we're consistently having problems, reset the worker
                    resetWorker();
                    consecutiveErrors = 0;
                }
            }
        }
        
        // Always render the current state
        gl.clear(gl.COLOR_BUFFER_BIT);
        if (vertexCount > 0) {
            gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
        }
    };
    
    // Initialize the first frame
    try {
        currentFrameProcessing = true;
        
        if (messageEl) messageEl.innerText = "Preparing first frame...";
        
        // Create initial buffer
        const initialBuffer = new ArrayBuffer(staticBytes.length + dynamicRaw[0].length);
        const initialData = new Uint8Array(initialBuffer);
        initialData.set(staticBytes);
        initialData.set(dynamicRaw[0], staticBytes.length);
        
        // Important: Send the view matrix first to ensure the worker has it
        const viewProj = multiply4(projectionMatrix, viewMatrix);
        worker.postMessage({ 
            view: viewProj,
            frameId: 0
        });
        
        // Then send the buffer - use a copy instead of transferring for the first frame
        // to ensure we don't have race conditions with view matrix
        worker.postMessage({ 
            buffer: initialData.buffer.slice(0), // Send a copy instead of transferring
            vertexCount: initialData.length / ROW_LEN,
            frameId: 0
        });
        
        if (messageEl) messageEl.innerText = "First frame sent to worker...";
    } catch (err) {
        console.error("Error initializing first frame:", err);
        if (messageEl) messageEl.innerText = "Error: " + err.message;
        currentFrameProcessing = false;
    }
    
    // Start the loop with a tracked animation frame
    currentAnimationFrame = requestAnimationFrame(loop);
}

// Lazy loading: only start canvas when button is clicked
let canvasLoaded = false;

// Function to update scene buttons state
function updateSceneButtonsState(selectedScene) {
    const buttons = document.querySelectorAll('.canvas-load-button');
    buttons.forEach(btn => {
        if (btn.dataset.scene === selectedScene) {
            btn.style.opacity = '0.7';
            btn.style.pointerEvents = 'none';
        } else {
            btn.style.opacity = '1';
            btn.style.pointerEvents = 'auto';
        }
    });
}

// Modified initCanvas function to support scene selection
function initCanvas(sceneName = 'dumpling') {
    if (canvasLoaded && !sceneName) return;
    
    if (canvasLoaded && sceneName) {
        // If already loaded, just switch scene
        applySceneConfig(sceneName);
        updateSceneButtonsState(sceneName);
        return;
    }
    
    canvasLoaded = true;
    
    // Update UI elements but keep the image visible
    const canvas = document.getElementById("canvas");
    const message = document.getElementById("message");
    const spinner = document.getElementById("spinner");
    const instructions = document.getElementById("canvas-instructions");
    const buttonText = document.getElementById("canvas-button-text");
    
    // Update scene buttons state
    updateSceneButtonsState(sceneName);
    
    // Update text content
    if (buttonText) buttonText.innerText = "Loading Interactive Viewer...";
    if (instructions) instructions.innerText = "Interactive Viewer (Loading...)";
    
    // Show loading indicators
    if (message) {
        message.style.display = "block";
        message.innerText = `Loading ${sceneName} scene...`;
    }
    if (spinner) spinner.style.display = "block";
    
    // Clear the grey background and prepare for content
    if (canvas) {
        canvas.style.backgroundColor = 'transparent';
    }
    
    // Apply scene configuration
    applySceneConfig(sceneName);
}

// Set up button click handlers when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Get all scene buttons
    const sceneButtons = document.querySelectorAll('.canvas-load-button');
    
    // Add click handlers to each button
    sceneButtons.forEach(button => {
        button.addEventListener('click', function() {
            const sceneName = this.dataset.scene;
            initCanvas(sceneName);
        });
    });
});
