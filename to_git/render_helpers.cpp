#include <iostream>
#include <vector> // For vertex data

// Include GLFW for windowing and context
#include <GLFW/glfw3.h>
#include <glad/glad.h> 

// Vertex and Fragment Shader Sources (loaded from files or embedded)
const char* vertexShaderSource = R"GLSL(
#version 330 core
layout (location = 0) in vec2 a_position;
uniform mat3 u_matrix;
void main() {
    gl_Position = vec4( (u_matrix * vec3(a_position, 1.0)).xy, 0.0, 1.0 );
}
)GLSL";
const char* fragmentShaderSource = R"GLSL(
#version 330 core
uniform vec4 u_color;
out vec4 outColor;
void main() {
    outColor = u_color;
}
)GLSL";

// Helper function to create and compile shaders
GLuint createShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[1024];
        glGetShaderInfoLog(shader, 1024, NULL, infoLog);
        std::cerr << "ERROR::SHADER::COMPILATION_FAILED for type " << (type == GL_VERTEX_SHADER ? "VERTEX" : "FRAGMENT") << "\n" << infoLog << std::endl;
    }
    return shader;
}

// Helper function to create and link shader program
GLuint createProgram(GLuint vertexShader, GLuint fragmentShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar infoLog[1024];
        glGetProgramInfoLog(program, 1024, NULL, infoLog);
        std::cerr << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    return program;
}

// Helper to set up VAO and VBO for generic geometry
void setupGeometry(GLuint vao, GLuint vbo, const std::vector<float>& vertices) {
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Unbind VBO and VAO
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

// Vertex data
vector<float> setGeometryBall() {
    vector<float> ballVertices = {
        0.0f, 13.0f, 10.0f, 10.0f, 0.0f, 7.0f,
        0.0f, 7.0f, 10.0f, 10.0f, 2.0f, 4.0f,
        2.0f, 4.0f, 10.0f, 10.0f, 4.0f, 2.0f,
        4.0f, 2.0f, 10.0f, 10.0f, 7.0f, 0.0f,
        7.0f, 0.0f, 10.0f, 10.0f, 13.0f, 0.0f,
        13.0f, 0.0f, 10.0f, 10.0f, 16.0f, 2.0f,
        16.0f, 2.0f, 10.0f, 10.0f, 18.0f, 4.0f,
        18.0f, 4.0f, 10.0f, 10.0f, 20.0f, 7.0f,
        20.0f, 7.0f, 10.0f, 10.0f, 20.0f, 13.0f,
        20.0f, 13.0f, 10.0f, 10.0f, 18.0f, 16.0f,
        18.0f, 16.0f, 10.0f, 10.0f, 16.0f, 18.0f,
        16.0f, 18.0f, 10.0f, 10.0f, 13.0f, 20.0f,
        13.0f, 20.0f, 10.0f, 10.0f, 7.0f, 20.0f,
        7.0f, 20.0f, 10.0f, 10.0f, 4.0f, 18.0f,
        4.0f, 18.0f, 10.0f, 10.0f, 2.0f, 16.0f,
        2.0f, 16.0f, 10.0f, 10.0f, 0.0f, 13.0f
    };
    return ballVertices;
}

vector<float> setGeometryTrack() {
    vector<float> trackVertices = {
        // Left
        0.0f, 0.0f, 15.0f, 0.0f, 0.0f, 20.0f,
        0.0f, 20.0f, 15.0f, 0.0f, 15.0f, 20.0f,
        // Middle
        15.0f, 20.0f, 15.0f, 10.0f, 515.0f, 10.0f,
        515.0f, 10.0f, 15.0f, 20.0f, 515.0f, 20.0f,
        // Right
        515.0f, 20.0f, 530.0f, 20.0f, 530.0f, 0.0f,
        530.0f, 0.0f, 515.0f, 20.0f, 515.0f, 0.0f
    };
    return trackVertices;
}

vector<float> setGeometrySled() {
    vector<float> sledVertices = {
        // sled
        -15.0f, 0.0f, 15.0f, 
        0.0f, -15.0f, 10.0f,
        -15.0f, 10.0f, 15.0f, 
        0.0f, 15.0f, 10.0f
    };
    return sledVertices;
}

vector<float> setGeometryStick(float l) {
    vector<float> stickVertices = {
        // stick
        -2.0f, 0.0f, -2.0f, 
        l, 2.0f, 0.0f, // Assuming 'l' is the length
        2.0f, 0.0f, -2.0f,
        l, 2.0f, l
    };
    return stickVertices;
}
