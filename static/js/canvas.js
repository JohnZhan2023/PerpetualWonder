
let defaultViewMatrix = [-1, 0, 0, 0,
    0, -1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1];
// let defaultViewMatrix = [1,0,0,0,
//                             0,1,0,0,
//                             0,0,1,0,
//                             0,0,0,1];
let yaw = 0;   // Rotation around the Y-axis
let pitch = 0; // Rotation around the X-axis
let movement = [0, 0, 0]; // Movement vector initialized to 0,0,0
const render_width = 970;
const render_height = 485;
let bbox = [-9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999];
let radius = 9999;

const cameras = [
    {
        id: 0,
        position: [
            0, 0, 0   // +left, +up, +forward
        ],
        rotation: [
            [-1, 0, 0],
            [0., -1, 0],
            [0, 0, 1],
        ],
        fy: 960,
        fx: 960,
    }
];

function getProjectionMatrix(fx, fy, width, height) {
    const znear = 0.01;
    const zfar = 100;
    return [
        [(2 * fx) / width, 0, 0, 0],
        [0, -(2 * fy) / height, 0, 0],
        [0, 0, zfar / (zfar - znear), 1],
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
    // 6*4 + 4 + 4 = 8*4
    // XYZ - Position (Float32)
    // XYZ - Scale (Float32)
    // RGBA - colors (uint8)
    // IJKL - quaternion/rot (uint8)
    const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
    let lastProj = [];
    let depthIndex = new Uint32Array();
    let lastVertexCount = 0;

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

    function generateTexture() {
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);
        const u_buffer = new Uint8Array(buffer);

        var texwidth = 1024 * 2; // Set to your desired width
        var texheight = Math.ceil((2 * vertexCount) / texwidth); // Set to your desired height
        var texdata = new Uint32Array(texwidth * texheight * 4); // 4 components per pixel (RGBA)
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
        }

        self.postMessage({ texdata, texwidth, texheight }, [texdata.buffer]);
    }

    function runSort(viewProj) {
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);
        if (lastVertexCount == vertexCount) {
            let dot =
                lastProj[2] * viewProj[2] +
                lastProj[6] * viewProj[6] +
                lastProj[10] * viewProj[10];
            if (Math.abs(dot - 1) < 0.01) {
                return;
            }
        } else {
            generateTexture();
            lastVertexCount = vertexCount;
        }

        console.time("sort");
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

        console.timeEnd("sort");

        lastProj = viewProj;
        self.postMessage({ depthIndex, viewProj, vertexCount }, [
            depthIndex.buffer,
        ]);
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

    let sortRunning;
    self.onmessage = (e) => {
        if (e.data.buffer) {
            buffer = e.data.buffer;
            vertexCount = e.data.vertexCount;
        } else if (e.data.vertexCount) {
            vertexCount = e.data.vertexCount;
        } else if (e.data.view) {
            viewProj = e.data.view;
            throttledSort();
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
async function main() {
    const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
    let splatData

    let active_camera = JSON.parse(JSON.stringify(cameras[0]));  // deep copy

    const downsample =
        1.0;

    const worker = new Worker(
        URL.createObjectURL(
            new Blob(["(", createWorker.toString(), ")(self)"], {
                type: "application/javascript",
            }),
        ),
    );

    const canvas = document.getElementById("canvas");
    // const fps = document.getElementById("fps");
    // const focal_x = document.getElementById("focal-x");
    // const focal_y = document.getElementById("focal-y");
    // const inner_width = document.getElementById("inner-width");
    // const inner_height = document.getElementById("inner-height");

    let currentCameraIndex = 0;
    let projectionMatrix;

    const gl = canvas.getContext("webgl2", {
        antialias: false,
    });

    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(vertexShader));

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(fragmentShader));

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    gl.useProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS))
        console.error(gl.getProgramInfoLog(program));

    gl.disable(gl.DEPTH_TEST); // Disable depth testing

    // Enable blending
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(
        gl.ONE_MINUS_DST_ALPHA,
        gl.ONE,
        gl.ONE_MINUS_DST_ALPHA,
        gl.ONE,
    );
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

    // Theses are the data passed to the shader program
    const u_projection = gl.getUniformLocation(program, "projection");
    const u_viewport = gl.getUniformLocation(program, "viewport");
    const u_focal = gl.getUniformLocation(program, "focal");
    const u_view = gl.getUniformLocation(program, "view");

    // positions
    const triangleVertices = new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]);
    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW);
    const a_position = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(a_position);
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);

    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);

    var u_textureLocation = gl.getUniformLocation(program, "u_texture");
    gl.uniform1i(u_textureLocation, 0);

    const indexBuffer = gl.createBuffer();
    const a_index = gl.getAttribLocation(program, "index");
    gl.enableVertexAttribArray(a_index);
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
    gl.vertexAttribIPointer(a_index, 1, gl.INT, false, 0, 0);
    gl.vertexAttribDivisor(a_index, 1);

    const resize = () => {
        use_intrinsics(active_camera);

        // gl.uniform2fv(u_viewport, new Float32Array([innerWidth, innerHeight]));
        // gl.canvas.width = Math.round(innerWidth / downsample);
        // gl.canvas.height = Math.round(innerHeight / downsample);
        gl.uniform2fv(u_viewport, new Float32Array([render_width, render_height]));
        gl.canvas.width = Math.round(render_width / downsample);
        gl.canvas.height = Math.round(render_height / downsample);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

        update_displayed_info(active_camera);
    };

    const use_camera = (camera) => {
        use_intrinsics(camera);
        use_extrinsics(camera);
    };

    const use_intrinsics = (camera) => {
        gl.uniform2fv(u_focal, new Float32Array([camera.fx, camera.fy]));
        projectionMatrix = getProjectionMatrix(
            camera.fx,
            camera.fy,
            render_width,
            render_height,
        );
        gl.uniformMatrix4fv(u_projection, false, projectionMatrix);
    };

    const use_extrinsics = (camera) => {
        viewMatrix = getViewMatrix(camera);
        gl.uniformMatrix4fv(u_view, false, viewMatrix);
        yaw = 0;
        pitch = 0;
        movement = [0, 0, 0];
    };

    const update_displayed_info = (camera) => {
        // focal_x.innerText = "focal_x  " + camera.fx;
        // focal_y.innerText = "focal_y  " + camera.fy;
        // inner_width.innerText = "inner_width  " + innerWidth;
        // inner_height.innerText = "inner_height  " + innerHeight;
    };

    window.addEventListener("resize", resize);
    resize();

    worker.onmessage = (e) => {
        if (e.data.buffer) {
            splatData = new Uint8Array(e.data.buffer);
            const blob = new Blob([splatData.buffer], {
                type: "application/octet-stream",
            });
            const link = document.createElement("a");
            link.download = "model.splat";
            link.href = URL.createObjectURL(blob);
            document.body.appendChild(link);
            link.click();
        } else if (e.data.texdata) {
            const { texdata, texwidth, texheight } = e.data;
            // console.log(texdata)
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texParameteri(
                gl.TEXTURE_2D,
                gl.TEXTURE_WRAP_S,
                gl.CLAMP_TO_EDGE,
            );
            gl.texParameteri(
                gl.TEXTURE_2D,
                gl.TEXTURE_WRAP_T,
                gl.CLAMP_TO_EDGE,
            );
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

            gl.texImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RGBA32UI,
                texwidth,
                texheight,
                0,
                gl.RGBA_INTEGER,
                gl.UNSIGNED_INT,
                texdata,
            );
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, texture);
        } else if (e.data.depthIndex) {
            const { depthIndex, viewProj } = e.data;
            gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, depthIndex, gl.DYNAMIC_DRAW);
            vertexCount = e.data.vertexCount;
        }
    };

    let activeKeys = [];
    window.addEventListener("keydown", (e) => {
        if (document.activeElement != document.body) return;
        if (e.code === "KeyF") {
            active_camera.fx += 10; // Adjust 10 to your desired increment value
            active_camera.fy += 10; // Adjust 10 to your desired increment value
            use_intrinsics(active_camera);
            update_displayed_info(active_camera);
        }

        if (e.code === "KeyG") {
            active_camera.fx -= 10; // Adjust 10 to your desired decrement value
            active_camera.fy -= 10; // Adjust 10 to your desired decrement value
            use_intrinsics(active_camera);
            update_displayed_info(active_camera);
        }

        if (!activeKeys.includes(e.code)) activeKeys.push(e.code);
        if (/\d/.test(e.key)) {
            currentCameraIndex = parseInt(e.key)
            active_camera = JSON.parse(JSON.stringify(cameras[currentCameraIndex]));
            use_camera(active_camera);
            update_displayed_info(active_camera);
        }
    });

    window.addEventListener("keyup", (e) => {
        if (document.activeElement != document.body) return;
        activeKeys = activeKeys.filter((k) => k !== e.code);
    });

    window.addEventListener("blur", () => {
        activeKeys = [];
    });

    let jumpDelta = 0;
    let vertexCount = 0;

    let lastTime = 0;  // This variable holds the timestamp of the last frame
    let avgFps = 0;
    let start = 0;

    
    // New touch controls with sensitivity threshold
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
                    activeKeys = activeKeys.filter(key => key !== 'KeyI' && key !== 'KeyK');
                    if (deltaY < 0) activeKeys.push('KeyI');
                    else activeKeys.push('KeyK');
                } else {
                    // Horizontal movement
                    activeKeys = activeKeys.filter(key => key !== 'KeyJ' && key !== 'KeyL');
                    if (deltaX < 0) activeKeys.push('KeyJ');
                    else activeKeys.push('KeyL');
                }
            } else {
                // Single-finger touch for movement
                if (Math.abs(deltaY) > Math.abs(deltaX)) {
                    // Vertical movement
                    activeKeys = activeKeys.filter(key => key !== 'KeyW' && key !== 'KeyS');
                    if (deltaY < 0) activeKeys.push('KeyW');
                    else activeKeys.push('KeyS');
                } else {
                    // Horizontal movement
                    activeKeys = activeKeys.filter(key => key !== 'KeyA' && key !== 'KeyD');
                    if (deltaX < 0) activeKeys.push('KeyA');
                    else activeKeys.push('KeyD');
                }
            }
            
            touchStartY = touchEndY;
            touchStartX = touchEndX;
        }
    });

    canvas.addEventListener('touchend', (e) => {
        e.preventDefault();
        // Remove all touch-related keys when touch ends
        activeKeys = activeKeys.filter(key => !['KeyW', 'KeyS', 'KeyA', 'KeyD', 'KeyI', 'KeyK', 'KeyJ', 'KeyL'].includes(key));
        isTwoFingerTouch = false;
    });

    const frame = (now) => {
        // Calculate the time elapsed since the last frame
        const deltaTime = now - lastTime;
        const minFrameTime = 30;  // Minimum frame time in milliseconds

        // Check if the elapsed time is less than the minimum frame time
        if (deltaTime < minFrameTime) {
            // Calculate how much longer to wait
            const additionalTime = minFrameTime - deltaTime;
            // Use setTimeout to delay the next requestAnimationFrame call
            setTimeout(() => {
                requestAnimationFrame(frame);
            }, additionalTime);
        } 
        else 
        {
            requestAnimationFrame(frame);
        };

        // Update the lastTime to the current time
        lastTime = now;

        let inv = invert4(defaultViewMatrix);

        if (activeKeys.includes("KeyJ")) yaw -= 0.01;
        if (activeKeys.includes("KeyL")) yaw += 0.01;
        if (activeKeys.includes("KeyI")) pitch += 0.005;
        if (activeKeys.includes("KeyK")) pitch -= 0.005;

        if (activeKeys.includes("KeyQ")) {
            document.getElementById("message").innerText = 'movement:' + movement + 'yaw:' + yaw + 'pitch:' + pitch + 'focal:' + active_camera.fx;
        }

        pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, pitch));
        pitch = Math.max(bbox[4], Math.min(bbox[5], pitch));
        yaw = Math.max(bbox[6], Math.min(bbox[7], yaw));

        // Compute movement vector increment based on yaw
        movement_speed = 0.7;
        let dx = 0, dz = 0, dy = 0;
        // if (isTouching) {
        //     const touchDeltaX = touchEndX - touchStartX;
        //     const touchDeltaY = touchEndY - touchStartY;
            
        //     // Adjust these thresholds as needed for sensitivity
        //     if (touchDeltaY < -10) dz += 0.01 * movement_speed; // Up (W)
        //     if (touchDeltaY > 10) dz -= 0.01 * movement_speed; // Down (S)
        //     if (touchDeltaX < -10) dx -= 0.01 * movement_speed; // Left (A)
        //     if (touchDeltaX > 10) dx += 0.01 * movement_speed; // Right (D)
        // } else {
        //     // Existing keyboard controls
        //     if (activeKeys.includes("KeyW")) dz += 0.01 * movement_speed;
        //     if (activeKeys.includes("KeyS")) dz -= 0.01 * movement_speed;
        //     if (activeKeys.includes("KeyA")) dx -= 0.01 * movement_speed;
        //     if (activeKeys.includes("KeyD")) dx += 0.01 * movement_speed;
        // }
        if (activeKeys.includes("KeyW")) dz += 0.01 * movement_speed;
        if (activeKeys.includes("KeyS")) dz -= 0.01 * movement_speed;
        if (activeKeys.includes("KeyA")) dx -= 0.01 * movement_speed;
        if (activeKeys.includes("KeyD")) dx += 0.01 * movement_speed;
        // if (activeKeys.includes("KeyN")) dy -= 0.01;
        // if (activeKeys.includes("KeyM")) dy += 0.01;

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
        worker.postMessage({ view: viewProj });

        // const currentFps = 1000 / (now - lastFrame) || 0;
        // avgFps = avgFps * 0.9 + currentFps * 0.1;

        if (vertexCount > 0) {
            // document.getElementById("spinner").style.display = "none";
            gl.uniformMatrix4fv(u_view, false, actualViewMatrix);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
        } else {
            gl.clear(gl.COLOR_BUFFER_BIT);
            // document.getElementById("spinner").style.display = "";
            start = Date.now() + 2000;
        }

    };

    frame();

    const selectFile = (file) => {
        const fr = new FileReader();
        fr.onload = () => {
            splatData = new Uint8Array(fr.result);
            console.log("Loaded", Math.floor(splatData.length / rowLength));
            worker.postMessage({
                buffer: splatData,
                vertexCount: Math.floor(splatData.length / rowLength),
            });
        };
        fr.readAsArrayBuffer(file);
    };

    const preventDefault = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };
    document.addEventListener("dragenter", preventDefault);
    document.addEventListener("dragover", preventDefault);
    document.addEventListener("dragleave", preventDefault);
    document.addEventListener("drop", (e) => {
        e.preventDefault();
        e.stopPropagation();
        selectFile(e.dataTransfer.files[0]);
    });


    const buttons = document.querySelectorAll('.button, .button-img');
    let currentActiveButton = null;

    buttons.forEach(button => {
        button.addEventListener('click', function() {
            if (currentActiveButton !== this) {
                // Remove active class from all buttons
                buttons.forEach(btn => btn.classList.remove('active'));

                // Add active class to the clicked button
                this.classList.add('active');

                currentActiveButton = this;

                // Call the function to handle the active button change
                if (this.dataset.url) {
                    handleActiveButtonChange(this.dataset.url);
                } else {
                    handleEmptyButtonActivation();
                }
            }
        });
    });


    function handleEmptyButtonActivation() {
        splatData = new Uint8Array(0);
        worker.postMessage({
            buffer: splatData.buffer,
            vertexCount: 0,
        });
        active_camera = JSON.parse(JSON.stringify(cameras[0]));
        use_camera(active_camera);
        update_displayed_info(active_camera);
    }

    // Function to handle the active button change
    async function handleActiveButtonChange(activeButtonUrl) {
        splatData = new Uint8Array(0);
        worker.postMessage({
            buffer: splatData.buffer,
            vertexCount: 0,
        });

        console.log(`Active button changed to URL: ${activeButtonUrl}`);
        active_camera = JSON.parse(JSON.stringify(cameras[0]));
        use_camera(active_camera);
        update_displayed_info(active_camera);
        if (activeButtonUrl.includes('minecraft')) {
            movement = [0.3701745151902974,0,-2.9261931203450615];
            bbox = [-0.5, 0.64, -3.2, 0.8, -0.2, 0.16, -0.2, 0.3];
            active_camera.fx = active_camera.fy = 1220;
            radius = 9999;
            yaw = -0.13;
        }
        else if (activeButtonUrl.includes('city_hall_monet')) {
            movement = [0, 0, 0];
            bbox = [-9999, 9999, -9999, 9999, -0.1, 0.35, -9999, 9999];
            active_camera.fx = active_camera.fy = 960;
            radius = 0.13;
        }
        else if (activeButtonUrl.includes('city_hall')) {
            movement = [0, 0, 0];
            bbox = [-9999, 9999, -9999, 9999, -0.1, 0.35, -1.2, 2.7];
            active_camera.fx = active_camera.fy = 960;
            radius = 0.1;
        }
        else if (activeButtonUrl.includes('campus_2')) {
            movement = [0.014332685967334207, 0, -0.04401724574130725];   
            yaw = -0.11;
            bbox = [-0.1, 0.25, -0.2, 0.12, -0.01, 0.35, -1.6, 1.47];
            active_camera.fx = active_camera.fy = 960;
            radius = 9999;
        }
        else if (activeButtonUrl.includes('cathedral_2')) {
            movement = [0, 0, 0];
            bbox = [-9999, 9999, -9999, 9999, -0.2, 0.16, -0.5, 0.86];
            active_camera.fx = active_camera.fy = 960;
            radius = 0.1;
        }
        else if (activeButtonUrl.includes('zelda')) {
            movement = [0.0, 0, -3.2];
            bbox = [-0.2, 0.3, -3.2, 0.2, -0.2, 0.16, -0.12, 0.3];
            active_camera.fx = active_camera.fy = 1700;
            radius = 9999;
        }
        else if (activeButtonUrl.includes('venice')) {
            movement = [0, 0, 0];
            bbox = [-9999, 9999, -9999, 9999, -0.03, 0.3, -9999, 9999];
            active_camera.fx = active_camera.fy = 960;
            radius = 0.1;
        }
        else if (activeButtonUrl.includes('disney')) {
            movement = [0, 0, 0];
            bbox = [-9999, 9999, -9999, 9999, -0.1, 0.35, -9999, 9999];
            active_camera.fx = active_camera.fy = 900;
            radius = 0.15;
        }
        else {
            movement = [0, 0, 0];
            bbox = [-9999, 9999, -9999, 9999, -9999, 9999, -9999, 9999];  // minx, maxx, minz, maxz, minpitch, maxpitch, minyaw, maxyaw
            radius = 9999;
            active_camera.fx = active_camera.fy = 960;
        }
        use_intrinsics(active_camera);

        const url = new URL(activeButtonUrl);
        const req = await fetch(url, {
            mode: "cors", // no-cors, *cors, same-origin
            credentials: "omit", // include, *same-origin, omit
        });
        if (req.status != 200)
            throw new Error(req.status + " Unable to load " + req.url);
        const reader = req.body.getReader();
        splatData = new Uint8Array(req.headers.get("content-length"));
        console.log('Request content length', req.headers.get("content-length"))

        let bytesRead = 0;
        let lastVertexCount = -1;
        let stopLoading = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done || stopLoading) break;

            splatData.set(value, bytesRead);
            bytesRead += value.length;

            if (vertexCount > lastVertexCount) {
                worker.postMessage({
                    buffer: splatData.buffer,
                    vertexCount: Math.floor(bytesRead / rowLength),
                });
                console.log("Loaded", Math.floor(bytesRead / rowLength));
                lastVertexCount = vertexCount;
                console.log('Vertex count:', vertexCount)
            }
        }
        if (!stopLoading)
            worker.postMessage({
                buffer: splatData.buffer,
                vertexCount: Math.floor(bytesRead / rowLength),
            });
        }

}

main().catch((err) => {
    document.getElementById("message").innerText = err.toString();
});