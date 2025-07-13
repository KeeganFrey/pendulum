#include "math_helpers.cpp"
#include "render_helpers.cpp"
#include "inputs.cpp"
#include "physics_engine.cpp"

#include <chrono> // For std::chrono
#include <thread> // For std::this_thread

void drawObject(GLuint vao, const Transform& transform, const glm::vec4& color, GLsizei vertexCount);
void animationLoop();

GLFWwindow* window;
GLuint shaderProgram;
GLint u_matrix_location;
GLint u_color_location;

struct Transform{
    float angle;
    float tx;
    float ty;
    float sx;
    float sy;
    float ox;
    float oy;
};

//constants needed for the code
float l = 50.0f;

int main() {
    // 1. Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Configure GLFW for OpenGL 3.3 core profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // For macOS

    // 2. Create a Window
    const int screenWidth = 800; // Example width
    const int screenHeight = 600; // Example height
    window = glfwCreateWindow(screenWidth, screenHeight, "OpenGL Conversion", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Set up keyboard input callback
    glfwSetKeyCallback(window, keyCallback);

    // 3. Initialize GLAD (or GLEW)
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    GLuint vertexShader = createShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = createShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    shaderProgram = createProgram(vertexShader, fragmentShader);

    // Delete shaders after linking
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Get uniform locations
    u_matrix_location = glGetUniformLocation(shaderProgram, "u_matrix");
    u_color_location = glGetUniformLocation(shaderProgram, "u_color");

    // VAOs and VBOs
    GLuint ballVAO, ballVBO;
    GLuint trackVAO, trackVBO;
    GLuint sledVAO, sledVBO;
    GLuint stickVAO, stickVBO;

    // Generate VAOs and VBOs and configure attributes
    glGenVertexArrays(1, &ballVAO);
    glGenBuffers(1, &ballVBO);
    setupGeometry(ballVAO, ballVBO, setGeometryBall());

    glGenVertexArrays(1, &trackVAO);
    glGenBuffers(1, &trackVBO);
    setupGeometry(trackVAO, trackVBO, setGeometryTrack());

    glGenVertexArrays(1, &sledVAO);
    glGenBuffers(1, &sledVBO);
    setupGeometry(sledVAO, sledVBO, setGeometrySled());

    glGenVertexArrays(1, &stickVAO);
    glGenBuffers(1, &stickVBO);
    setupGeometry(stickVAO, stickVBO, setGeometryStick(l));

    glEnable(GL_DEPTH_TEST);

    //rendering loop
    animationLoop();

    // Clean up
    glDeleteVertexArrays(1, &ballVAO);
    glDeleteBuffers(1, &ballVBO);
    glDeleteVertexArrays(1, &trackVAO);
    glDeleteBuffers(1, &trackVBO);
    glDeleteVertexArrays(1, &sledVAO);
    glDeleteBuffers(1, &sledVBO);
    glDeleteVertexArrays(1, &stickVAO);
    glDeleteBuffers(1, &stickVBO);

    glDeleteProgram(shaderProgram);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

void drawObject(GLuint vao, const Transform& transform, const arma::vec color, GLsizei vertexCount) {
    glUseProgram(shaderProgram);

    // Set the color
    glUniform4fv(u_color_location, 1, color.memptr());

    // Get window dimensions for projection
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    // Calculate the model matrix
    arma::mat model = translation(transform.ox, transform.oy) * scale(transform.sx, transform.sy) * rotation(transform.angle) * translation(transform.tx, transform.ty) * projection(width, height);

    // Set the matrix uniform
    glUniformMatrix3fv(u_matrix_location, 1, GL_FALSE, model.memptr());

    // Bind the VAO and draw
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, vertexCount);
    glBindVertexArray(0); // Unbind VAO
}

struct Transform set_up_transform(float angle, float tx, float ty, float sx, float sy, float ox, float oy){
    struct Transform rvalue;
    rvalue.angle = angle;
    rvalue.tx = tx;
    rvalue.ty = ty;
    rvalue.sx = sx;
    rvalue.sy = sy;
    rvalue.ox = ox;
    rvalue.oy = oy;
    return rvalue;
}

void animationLoop(GLuint trackVAO, GLuint ballVAO, GLuint sledVAO, GLuint stickVAO) {
    arma::vec c1 = arma::randu<arma::vec>(4);
    arma::vec c2 = arma::randu<arma::vec>(4);
    arma::vec c3 = arma::randu<arma::vec>(4);
    arma::vec c4 = arma::randu<arma::vec>(4);

    double dt = .01;

    struct Transform trackTransform = set_up_transform(315, 180, 0, 1, 1, -265, -10);
    struct Transform ballTransform = set_up_transform(315, 170, 0, .5, .5, 0, 0);
    struct Transform sledTransform = set_up_transform(315, 170, 0, 1, 1, 0, 0);
    struct Transform stickTransform = set_up_transform(315, 170, 0, 1, 1, 0, 0);

    sled_state[0] = 315;
    sled_state[1] = 170;
    pend_state[2] = 315;
    pend_state[3] = 170;

    while (!glfwWindowShouldClose(window)) {
        // Clear buffers
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear color and depth buffers

        // Apply physics for the current frame
        run_step(sled_state, pend_state, output, dt);

        sledTransform.tx = sled_state[0];
        sledTransform.ty = sled_state[1];

        stickTransform.tx = pend_state[2];
        stickTransform.ty = pend_state[3];
        stickTransform.angle = pend_state[0];

        ballTransform.tx = pend_state[2] + l * cos(pend_state[0]);
        ballTransform.ty = pend_state[3] + l * sin(pend_state[0]);

        // Draw all objects
        // Note: vertex counts are (number of vertices) / (components per vertex).
        // Since each vertex is vec2, each triangle is 3 * 2 = 6 floats.
        drawObject(trackVAO, trackTransform, c1, 6 * 3); // Track has 6 triangles
        drawObject(ballVAO, ballTransform, c2, 16 * 3);  // Ball has 16 triangles
        drawObject(sledVAO, sledTransform, c3, 2 * 3);   // Sled has 2 triangles
        drawObject(stickVAO, stickTransform, c4, 2 * 3); // Stick has 2 triangles

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();

        std::cout << "Program will sleep for " << dt*100 << " seconds..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(dt*100));
        std::cout << "Program resumed." << std::endl;
    }
}
