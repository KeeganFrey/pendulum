"use strict";

var vertexShaderSource = `#version 300 es

// an attribute is an input (in) to a vertex shader.
// It will receive data from a buffer
in vec2 a_position;

// A matrix to transform the positions by
uniform mat3 u_matrix;

// all shaders have a main function
void main() {
  // Multiply the position by the matrix.
  gl_Position = vec4((u_matrix * vec3(a_position, 1)).xy, 0, 1);
}
`;

var fragmentShaderSource = `#version 300 es

precision highp float;

uniform vec4 u_color;

// we need to declare an output for the fragment shader
out vec4 outColor;

void main() {
  outColor = u_color;
}
`;

var m3 = {
  projection: function projection(width, height) {
    // Note: This matrix flips the Y axis so that 0 is at the top.
    return [
      2 / width, 0, 0,
      0, -2 / height, 0,
      -1, 1, 1,
    ];
  },

  translation: function translation(tx, ty) {
    return [
      1, 0, 0,
      0, 1, 0,
      tx, ty, 1,
    ];
  },

  rotation: function rotation(angleInRadians) {
    var c = Math.cos(angleInRadians);
    var s = Math.sin(angleInRadians);
    return [
      c, -s, 0,
      s, c, 0,
      0, 0, 1,
    ];
  },

  scaling: function scaling(sx, sy) {
    return [
      sx, 0, 0,
      0, sy, 0,
      0, 0, 1,
    ];
  },

  multiply: function multiply(a, b) {
    var a00 = a[0 * 3 + 0];
    var a01 = a[0 * 3 + 1];
    var a02 = a[0 * 3 + 2];
    var a10 = a[1 * 3 + 0];
    var a11 = a[1 * 3 + 1];
    var a12 = a[1 * 3 + 2];
    var a20 = a[2 * 3 + 0];
    var a21 = a[2 * 3 + 1];
    var a22 = a[2 * 3 + 2];
    var b00 = b[0 * 3 + 0];
    var b01 = b[0 * 3 + 1];
    var b02 = b[0 * 3 + 2];
    var b10 = b[1 * 3 + 0];
    var b11 = b[1 * 3 + 1];
    var b12 = b[1 * 3 + 2];
    var b20 = b[2 * 3 + 0];
    var b21 = b[2 * 3 + 1];
    var b22 = b[2 * 3 + 2];
    return [
      b00 * a00 + b01 * a10 + b02 * a20,
      b00 * a01 + b01 * a11 + b02 * a21,
      b00 * a02 + b01 * a12 + b02 * a22,
      b10 * a00 + b11 * a10 + b12 * a20,
      b10 * a01 + b11 * a11 + b12 * a21,
      b10 * a02 + b11 * a12 + b12 * a22,
      b20 * a00 + b21 * a10 + b22 * a20,
      b20 * a01 + b21 * a11 + b22 * a21,
      b20 * a02 + b21 * a12 + b22 * a22,
    ];
  },

  translate: function(m, tx, ty) {
    return m3.multiply(m, m3.translation(tx, ty));
  },

  rotate: function(m, angleInRadians) {
    return m3.multiply(m, m3.rotation(angleInRadians));
  },

  scale: function(m, sx, sy) {
    return m3.multiply(m, m3.scaling(sx, sy));
  },
};

let height=0;
let width=0;
let f_i = 0;
let l = 50;

// Get A WebGL context
let canvas = document.querySelector("#c");
height = canvas.clientHeight;
width = canvas.clientWidth;
let gl = canvas.getContext("webgl2");
if (!gl) {
  console.log("Error getting gl");
}
gl.canvas.width = width;
gl.canvas.height = height;
let vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
let fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
let program = createProgram(gl, vertexShader, fragmentShader);

// look up where the vertex data needs to go.
let positionAttributeLocation = gl.getAttribLocation(program, "a_position");

// look up uniform locations
let colorLocation = gl.getUniformLocation(program, "u_color");
let matrixLocation = gl.getUniformLocation(program, "u_matrix");

//General values for defining the geometry
let size = 2;          // 2 components per iteration
let type = gl.FLOAT;   // the data is 32bit floats
let normalize = false; // don't normalize the data
let stride = 0;        // 0 = move forward size * sizeof(type) each iteration to get the next position
let offset = 0;        // start at the beginning of the buffer

//End of setup

let identity = m3.rotation(0);

//Start of Ball creation
// Create a buffer
let positionBufferb = gl.createBuffer();
let vaob = gl.createVertexArray();
// and make it the one we're currently working with
gl.bindVertexArray(vaob);
// Turn on the attribute
gl.enableVertexAttribArray(positionAttributeLocation);
// Bind it to ARRAY_BUFFER (think of it as ARRAY_BUFFER = positionBuffer)
gl.bindBuffer(gl.ARRAY_BUFFER, positionBufferb);
setGeometryBall(gl);
gl.vertexAttribPointer(
    positionAttributeLocation, size, type, normalize, stride, offset);
let translationb = [200, 150];
let rotationInRadiansb = 0;
let scaleb = [.5, .5];
let colorb = [Math.random(), Math.random(), Math.random(), 1];
//end of Ball creation

//Start of Track creation
// Create a buffer
let positionBuffert = gl.createBuffer();
let vaot = gl.createVertexArray();
// and make it the one we're currently working with
gl.bindVertexArray(vaot);
// Turn on the attribute
gl.enableVertexAttribArray(positionAttributeLocation);
// Bind it to ARRAY_BUFFER (think of it as ARRAY_BUFFER = positionBuffer)
gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffert);
setGeometryTrack(gl);
gl.vertexAttribPointer(
    positionAttributeLocation, size, type, normalize, stride, offset);
let translationt = [315, 180];
let rotationInRadianst = 0;
let scalet = [1, 1];
let movorigin = m3.translation(-265, -10)
let colort = [Math.random(), Math.random(), Math.random(), 1];
//end of Track creation

//Start of sled creation
// Create a buffer
let positionBuffers = gl.createBuffer();
let vaos = gl.createVertexArray();
// and make it the one we're currently working with
gl.bindVertexArray(vaos);
// Turn on the attribute
gl.enableVertexAttribArray(positionAttributeLocation);
// Bind it to ARRAY_BUFFER (think of it as ARRAY_BUFFER = positionBuffer)
gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffers);
setGeometrySled(gl);
gl.vertexAttribPointer(
    positionAttributeLocation, size, type, normalize, stride, offset);
let translations = [315, 170];
let rotationInRadianss = 0;
let scales = [1, 1];
let colors = [Math.random(), Math.random(), Math.random(), 1];
//end of sled creation

//Start of stick creation
// Create a buffer
let positionBufferst = gl.createBuffer();
let vaost = gl.createVertexArray();
// and make it the one we're currently working with
gl.bindVertexArray(vaost);
// Turn on the attribute
gl.enableVertexAttribArray(positionAttributeLocation);
// Bind it to ARRAY_BUFFER (think of it as ARRAY_BUFFER = positionBuffer)
gl.bindBuffer(gl.ARRAY_BUFFER, positionBufferst);
setGeometryStick(gl);
gl.vertexAttribPointer(
    positionAttributeLocation, size, type, normalize, stride, offset);
let translationst = [315, 170];
let rotationInRadiansst = 0;
let scalest = [1, 1];
let colorst = [Math.random(), Math.random(), Math.random(), 1];
//end of sled creation

drawScene();

function setGeometrySled(gl) {
  gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([
          //sled
          -15, 0,
          15, 0,
          -15, 10,

          -15, 10,
          15, 0,
          15, 10,
      ]),
      gl.STATIC_DRAW);
}

function setGeometryStick(gl) {
  gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([
          //sled
          -2, 0,
          -2, l,
          2, 0,

          2, 0,
          2, l,
          -2, l,
      ]),
      gl.STATIC_DRAW);
}

function setGeometryTrack(gl) {
  gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([
          //left
          0, 0,
          15, 0,
          0, 20,

          0, 20,
          15, 0,
          15, 20,
          //middle
          15,20,
          15,10,
          515,10,

          515,10,
          15,20,
          515,20,
          //right
          515, 20,
          530, 20,
          530, 0,

          530, 0,
          515, 20,
          515, 0,
      ]),
      gl.STATIC_DRAW);
}

function setGeometryBall(gl) {
  gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([
          // middle rung
          0, 13,
          10, 10,
          0, 7,

          0, 7,
          10, 10,
          2, 4,

          2, 4,
          10, 10,
          4, 2,

          4, 2,
          10, 10,
          7, 0,
          //5
          7, 0,
          10, 10,
          13, 0,

          13, 0,
          10, 10,
          16, 2,

          16, 2,
          10, 10,
          18, 4,

          18, 4,
          10, 10,
          20, 7,

          20, 7,
          10, 10,
          20, 13,
          //10
          20, 13,
          10, 10,
          18, 16,

          18, 16,
          10, 10,
          16, 18,
          
          16, 18,
          10, 10,
          13, 20, 

          13, 20,
          10, 10,
          7, 20,

          7, 20,
          10, 10,
          4, 18,
          //15
          4, 18,
          10, 10,
          2, 16,

          2, 16,
          10, 10,
          0, 13,
      ]),
      gl.STATIC_DRAW);
}

function createShader(gl, type, source) {
  var shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  var success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
  if (success) {
    return shader;
  }

  console.log(gl.getShaderInfoLog(shader));  // eslint-disable-line
  gl.deleteShader(shader);
  return undefined;
}

function createProgram(gl, vertexShader, fragmentShader) {
  var program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  var success = gl.getProgramParameter(program, gl.LINK_STATUS);
  if (success) {
    return program;
  }

  console.log(gl.getProgramInfoLog(program));  // eslint-disable-line
  gl.deleteProgram(program);
  return undefined;
}

function drawObject(vao, colorloc, color, translte, rot, scl, m_loc, cnt, orgi){
    gl.bindVertexArray(vao);

    // Set the color.
    gl.uniform4fv(colorloc, color);

    let matrix = m3.projection(gl.canvas.clientWidth, gl.canvas.clientHeight);
    // Compute the matrix
    matrix = m3.translate(matrix, translte[0], translte[1]);
    // Then apply the rotation around that pivot
    matrix = m3.rotate(matrix, rot);

    // Then apply the scale (if any)
    matrix = m3.scale(matrix, scl[0], scl[1]);

    // Finally, translate the scaled and rotated object to its final position
    matrix = m3.multiply(matrix, orgi); // Apply the pivot translation first
    

    // Set the matrix.
    gl.uniformMatrix3fv(m_loc, false, matrix);

    // Draw the geometry.
    let primitiveType = gl.TRIANGLES;
    let offset = 0;
    gl.drawArrays(primitiveType, offset, cnt);
}

// Draw the scene.
  function drawScene() {
    // Tell WebGL how to convert from clip space to pixels
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    // Clear the canvas
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // Tell it to use our program (pair of shaders)
    gl.useProgram(program);

    //Start of drawing one object
    //drawObject(vaof, colorLocation, colorf, translationf, rotationInRadiansf, scalef, matrixLocation, 18);
    drawObject(vaot, colorLocation, colort, translationt, rotationInRadianst, scalet, matrixLocation, 6 * 3, movorigin);
    drawObject(vaob, colorLocation, colorb, translationb, rotationInRadiansb, scaleb, matrixLocation, 16 * 3, identity); 
    drawObject(vaos, colorLocation, colors, translations, rotationInRadianss, scales, matrixLocation, 2 * 3, identity);  
    drawObject(vaost, colorLocation, colorst, translationst, rotationInRadiansst, scalest, matrixLocation, 2 * 3, identity); 
  }

/*******************************/
/* Button and keypress handler */
/*******************************/

// --- Internal Variables ---
let isFeatureEnabled = false; // Our internal state variable for the toggle
let currentNumber = 0;       // Our number to be incremented/decremented
let currentNumberx = 0;       // Our number to be incremented/decremented (controls X translation)
let currentNumbery = 0;       // Our number to be incremented/decremented (controls Y translation)
let currentNumberz = 0;       // Our number to be incremented/decremented (controls rotation)

// --- DOM Element References ---
const inputDisplay = document.getElementById('inputDisplay');
const counterDisplay = document.getElementById('counterDisplay'); // Generic counter
const counterDisplayx = document.getElementById('counterDisplayx'); // Not strictly used now, but good to have
const counterDisplayy = document.getElementById('counterDisplayy'); // Not strictly used now, but good to have
const counterDisplayz = document.getElementById('counterDisplayz'); // Not strictly used now, but good to have

// *** NEW: References for the new display elements ***
const counterDisplayFi = document.getElementById('counterDisplayFi');
const counterDisplayXValue = document.getElementById('counterDisplayXValue');
const counterDisplayThVValue = document.getElementById('counterDisplayTVValue');
const counterDisplayThAValue = document.getElementById('counterDisplayTAValue');

const toggleButton = document.getElementById('toggleButton');
const statusDisplay = document.getElementById('statusDisplay');

// --- References to the Key Indicator Buttons ---
const wKeyIndicator = document.getElementById('wKeyIndicator');
const aKeyIndicator = document.getElementById('aKeyIndicator');
const sKeyIndicator = document.getElementById('sKeyIndicator');
const dKeyIndicator = document.getElementById('dKeyIndicator');
const spaceKeyIndicator = document.getElementById('spaceKeyIndicator');
const shiftKeyIndicator = document.getElementById('shiftKeyIndicator');

// Helper to update indicator button styles
function updateKeyIndicator(keyElement, isPressed) {
    if (keyElement) {
        if (isPressed) {
            keyElement.classList.add('pressed');
            keyElement.classList.remove('not-pressed');
        } else {
            keyElement.classList.add('not-pressed');
            keyElement.classList.remove('pressed');
        }
    }
}

// --- Event Listener for Keyboard Input ---
document.addEventListener('keydown', function(event) {
    const keyPressed = event.key;
    const keyCode = event.code;
    const shiftKey = event.shiftKey;
    const ctrlKey = event.ctrlKey;
    const altKey = event.altKey;

    // Update the input display
    inputDisplay.textContent = `Key: "${keyPressed}" | Code: "${keyCode}" | Shift: ${shiftKey} | Ctrl: ${ctrlKey} | Alt: ${altKey}`;

    // --- Logic for Incrementing/Decrementing and setting f_i ---
    if (keyPressed === '+') {
        if (!shiftKey) {
            currentNumber++;
        } else {
             currentNumber++; // Assuming shift+plus does the same as plus for generic
        }
        updateCounterDisplay(0); // Update generic counter
    } else if (keyPressed === '-') {
        currentNumber--;
        updateCounterDisplay(0); // Update generic counter
    }
    // Mapping keys to transformations and setting f_i
    else if (keyPressed === 'w') {
        currentNumberz++; // Rotate counter-clockwise
        updateKeyIndicator(wKeyIndicator, true);
    }
    else if (keyPressed === 's') {
        currentNumberz--; // Rotate clockwise
        updateKeyIndicator(sKeyIndicator, true);
    }
    else if (keyPressed === 'a') {
        currentNumberx--; // Move left
        // *** MODIFIED: Set f_i to -10 when 'a' is pressed ***
        f_i = -100;
        updateKeyIndicator(aKeyIndicator, true);
    }
    else if (keyPressed === 'd') {
        currentNumberx++; // Move right
        // *** MODIFIED: Set f_i to 10 when 'd' is pressed ***
        f_i = 100;
        updateKeyIndicator(dKeyIndicator, true);
    }
    else if (keyPressed === ' ') { // Spacebar
        currentNumbery++; // Move up
        updateKeyIndicator(spaceKeyIndicator, true);
    }
    else if (keyPressed === "Shift") { // Shift key
        currentNumbery--; // Move down
        updateKeyIndicator(shiftKeyIndicator, true);
    }

    // After updating any counter, redraw the scene to reflect changes
    drawScene();
    // Update the f_i and X value displays
    updateSpecificDisplays();
});

// --- Event Listener for Keyboard Release ---
document.addEventListener('keyup', function(event) {
    const keyPressed = event.key;

    // Reset the indicator for the released key
    if (keyPressed === 'w') updateKeyIndicator(wKeyIndicator, false);
    else if (keyPressed === 's') updateKeyIndicator(sKeyIndicator, false);
    else if (keyPressed === 'a') {
        updateKeyIndicator(aKeyIndicator, false);
        // *** MODIFIED: Reset f_i to 0 when 'a' is released ***
        f_i = 0;
    }
    else if (keyPressed === 'd') {
        updateKeyIndicator(dKeyIndicator, false);
        // *** MODIFIED: Reset f_i to 0 when 'd' is released ***
        f_i = 0;
    }
    else if (keyPressed === ' ') updateKeyIndicator(spaceKeyIndicator, false);
    else if (keyPressed === "Shift") updateKeyIndicator(shiftKeyIndicator, false);

    // Update the f_i and X value displays
    updateSpecificDisplays();
});


// --- Helper Function to Update Counter Displays ---
function updateCounterDisplay(dim) {
    if (dim === 0) { // Generic counter
        counterDisplay.textContent = `Generic Counter: ${currentNumber}`;
    }
}

//z vector, z1=θ, z2=θ', z3=x, z4=y, z5=x', z6=y' 
let z = [0,0,translationb[0],translationb[1],0,0]
//z vector, z1=θ', z2=θ'', z3=x', z4=y', z5=x_s'', z6=y_s'' 
let F = [0,0,0,0,0,0]
let dx_s = 0;

// *** NEW HELPER FUNCTION to update the f_i and X displays ***
function updateSpecificDisplays() {
    counterDisplayFi.textContent = `f_i: ${f_i}`;
    counterDisplayXValue.textContent = `X Velocity: ${dx_s}`;
    counterDisplayThVValue.textContent = `Theta Velocity: ${z[1]}`; 
    counterDisplayThAValue.textContent = `Theta Acc: ${F[1]}`; 
}


// --- Initial Setup ---
function initializeUI() {
    // Initialize the displays with their starting values
    updateCounterDisplay(0); // Set the initial counter display for generic
    updateSpecificDisplays(); // Set the initial f_i and X displays

    // Initialize indicator buttons to not pressed (red)
    updateKeyIndicator(wKeyIndicator, false);
    updateKeyIndicator(aKeyIndicator, false);
    updateKeyIndicator(sKeyIndicator, false);
    updateKeyIndicator(dKeyIndicator, false);
    updateKeyIndicator(spaceKeyIndicator, false);
    updateKeyIndicator(shiftKeyIndicator, false);

    // Set initial toggle button state
    if (isFeatureEnabled) {
        toggleButton.textContent = 'Toggle State (On)';
        toggleButton.classList.remove('off');
        statusDisplay.textContent = 'Current State: true';
    } else {
        toggleButton.textContent = 'Toggle State (Off)';
        toggleButton.classList.add('off');
        statusDisplay.textContent = 'Current State: false';
    }
}

/***********************************/
/* End Button and keypress handler */
/***********************************/

let x_s = 0;
let y_s = 0;
let dy_s = 0;
let dx2_s = 0;
let dy2_s = 0;
let b = .5;
let b_t = .1;
let delta_t = 10;
let g = 400;

function applyPhysics(){
  dx2_s = f_i - b * dx_s;
  dx_s += dx2_s * delta_t/1000;
  translations[0] += dx_s * delta_t/1000;

  if(translations[0] < translationt[0]-235 || translations[0] > translationt[0]+235){
    dx_s = -dx_s;
    translations[0] += dx_s * delta_t/1000;
    translations[0] += dx_s * delta_t/1000;

    dx2_s = 1000 * dx_s / delta_t;
  }

  if(dx_s < 1 && dx_s > -1){
    dx_s = 0;
  }

  F[0] = z[1];
  F[1] = (-1/l)*(dx2_s * Math.cos(z[0]) + (dy2_s + g) * Math.sin(z[0])) - b_t * z[1];
  F[2] = z[4]
  F[3] = z[5]
  F[4] = dx2_s;
  F[5] = dy2_s;

  for(let i = 0; i < 6; i++){
    z[i] = z[i] + F[i] * delta_t / 1000;
  }

  rotationInRadiansst = z[0];
  translationst[0] = translations[0];
  translationst[1] = translations[1]; 
  translationb[0] = translationst[0] + l * Math.sin(rotationInRadiansst) - 5;
  translationb[1] = translationst[1] + l * Math.cos(rotationInRadiansst) - 5;

  drawScene();
  updateSpecificDisplays();
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function loop() {
  while(true){
    await sleep(delta_t);
    applyPhysics();
  }
}

loop();
