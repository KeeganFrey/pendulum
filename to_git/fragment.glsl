#version 330 core // Use a modern desktop GLSL version

// Uniforms are the same.
uniform vec4 u_color;

// The output color for the fragment.
// In desktop GLSL, you can still use gl_FragColor if you don't
// explicitly declare an output, but declaring it is good practice.
out vec4 outColor;

void main() {
    outColor = u_color;
}