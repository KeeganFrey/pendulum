#version 330 core // Use a modern desktop GLSL version

// "layout" specifies the binding point for the attribute.
// This corresponds to positionAttributeLocation in WebGL.
layout (location = 0) in vec2 a_position;

// A matrix to transform the positions by.
uniform mat3 u_matrix;

void main() {
    // Multiply the position by the matrix.
    // In GLSL, matrices are column-major.
    // The input a_position is vec2, so we make it vec3 for multiplication
    // by appending a '1' for the W component (homogeneous coordinates).
    // The result of (u_matrix * vec3(a_position, 1)) is a vec3.
    // We take its xy components for gl_Position and set z=0, w=1.
    gl_Position = vec4( (u_matrix * vec3(a_position, 1.0)).xy, 0.0, 1.0 );
}