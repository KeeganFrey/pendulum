// Include GLFW for windowing and context
#include <GLFW/glfw3.h>
#include <glad/glad.h> 

GLFWwindow* window;
GLuint shaderProgram;
GLint u_matrix_location;
GLint u_color_location;

void drawObject(GLuint vao, const Transform& transform, const arma::vec color, GLsizei vertexCount, int renderWidth, int renderHeight);
void animationLoop(GLuint trackVAO, GLuint ballVAO, GLuint sledVAO, GLuint stickVAO);
void setupOffscreenFramebuffer();
void cleanupOffscreenFramebuffer();

const unsigned int OFFSCREEN_WIDTH = 800, OFFSCREEN_HEIGHT = 600;
GLuint fbo; // Framebuffer Object
GLuint textureColorbuffer; // The texture we will render to
GLuint rbo; // Renderbuffer Object for depth/stencil tests

// A buffer to hold the pixel data read from the GPU
unsigned char* pixel_data_buffer = new unsigned char[OFFSCREEN_WIDTH * OFFSCREEN_HEIGHT * 3];

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
    // NOTE: A window is still required to create an OpenGL context, even for off-screen rendering.
    // You could later make this window hidden if you don't need a visible display.
    const int screenWidth = 800;
    const int screenHeight = 600;
    window = glfwCreateWindow(screenWidth, screenHeight, "OpenGL Off-Screen Processing", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Set up keyboard input callback
    glfwSetKeyCallback(window, keyCallback);

    // 3. Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        glfwDestroyWindow(window); // Clean up window on failure
        glfwTerminate();
        return -1;
    }

    // --- Shader and Uniform Setup (No Changes) ---
    GLuint vertexShader = createShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = createShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    shaderProgram = createProgram(vertexShader, fragmentShader);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    u_matrix_location = glGetUniformLocation(shaderProgram, "u_matrix");
    u_color_location = glGetUniformLocation(shaderProgram, "u_color");

    // --- Geometry Setup (No Changes) ---
    GLuint ballVAO, ballVBO;
    GLuint trackVAO, trackVBO;
    GLuint sledVAO, sledVBO;
    GLuint stickVAO, stickVBO;

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

    setupOffscreenFramebuffer();

    // NOTE: You don't necessarily need to enable depth test for the main context
    // if you only render to the FBO, but it doesn't hurt. It's enabled inside
    // the animation loop for the FBO context.
    glEnable(GL_DEPTH_TEST);

    // --- MODIFIED: Call animationLoop with VAO handles ---
    // The animation loop now needs handles to the objects it's supposed to draw.
    animationLoop(trackVAO, ballVAO, sledVAO, stickVAO);

    cleanupOffscreenFramebuffer();

    // --- Cleanup ---
    // The cleanup order is important. Delete your geometry first.
    glDeleteVertexArrays(1, &ballVAO);
    glDeleteBuffers(1, &ballVBO);
    glDeleteVertexArrays(1, &trackVAO);
    glDeleteBuffers(1, &trackVBO);
    glDeleteVertexArrays(1, &sledVAO);
    glDeleteBuffers(1, &sledVBO);
    glDeleteVertexArrays(1, &stickVAO);
    glDeleteBuffers(1, &stickVBO);

    // Delete the shader program
    glDeleteProgram(shaderProgram);

    // Finally, destroy the window and terminate GLFW
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}



// MODIFIED drawObject function
// It now takes width and height as arguments instead of querying the window.
void drawObject(GLuint vao, const Transform& transform, const arma::vec color, GLsizei vertexCount, int renderWidth, int renderHeight) {
    glUseProgram(shaderProgram);

    glUniform4fv(u_color_location, 1, color.memptr());

    // Calculate the model matrix using the provided dimensions
    arma::mat model = translation(transform.ox, transform.oy) * scale(transform.sx, transform.sy) * rotation(transform.angle) * translation(transform.tx, transform.ty) * projection(renderWidth, renderHeight);

    glUniformMatrix3fv(u_matrix_location, 1, GL_FALSE, model.memptr());

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, vertexCount);
    glBindVertexArray(0);
}





// MODIFIED animationLoop
void animationLoop(GLuint trackVAO, GLuint ballVAO, GLuint sledVAO, GLuint stickVAO) {

    // (Your existing setup code for colors, transforms, etc. remains here)
    arma::vec c1 = arma::randu<arma::vec>(4);
    // ... same for c2, c3, c4
    struct Transform trackTransform = set_up_transform(315, 180, 0, 1, 1, -265, -10);
    // ... etc.

    while (!glfwWindowShouldClose(window)) {
        // --- RENDER TO OFF-SCREEN FBO ---
        // 1. Bind our custom framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT); // Use the offscreen buffer's dimensions

        // 2. Clear the attached buffers
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Apply physics for the current frame
        run_step(sled_state, pend_state, output, dt);

        sledTransform.tx = sled_state[0];
        sledTransform.ty = sled_state[1];

        stickTransform.tx = pend_state[2];
        stickTransform.ty = pend_state[3];
        stickTransform.angle = pend_state[0];

        ballTransform.tx = pend_state[2] + l * cos(pend_state[0]);
        ballTransform.ty = pend_state[3] + l * sin(pend_state[0]);

        // 3. Draw all objects to the FBO, passing the offscreen dimensions
        drawObject(trackVAO, trackTransform, c1, 6 * 3, OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT);
        drawObject(ballVAO, ballTransform, c2, 16 * 3, OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT);
        drawObject(sledVAO, sledTransform, c3, 2 * 3, OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT);
        drawObject(stickVAO, stickTransform, c4, 2 * 3, OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT);

        // --- GET FRAME AND PROCESS ---
        glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, pixel_data_buffer);

        // 5. Call your processing function with the data
        processFrameData(pixel_data_buffer, OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT);


        // --- FINALIZE FRAME ---
        glBindFramebuffer(GL_FRAMEBUFFER, 0); //Unbind the FBO to switch back to the default window buffer

        // Since nothing was rendered to the default buffer, the screen will be blank or whatever
        // was there before. This is expected and correct for your use case.
        // If you wanted to see the result for debugging, you would now draw a full-screen quad
        // with the 'textureColorbuffer' applied to it before swapping buffers.

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

}

// --- New FBO Setup Function ---
void setupOffscreenFramebuffer() {
    // 1. Create and bind the Framebuffer Object (FBO)
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // 2. Create a texture to use as the color attachment
    glGenTextures(1, &textureColorbuffer);
    glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
    // Allocate memory for the texture. We pass NULL for the data parameter because we are about to render to it.
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // Attach the texture to the FBO
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);

    // 3. Create a Renderbuffer Object (RBO) for depth and stencil attachment
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    // Use a single renderbuffer object for both a depth AND stencil buffer.
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT);
    // Attach the renderbuffer object to the FBO
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

    // 4. Check if the framebuffer is complete
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
    }

    // 5. Unbind the FBO to return to the default framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// --- New FBO Cleanup Function ---
void cleanupOffscreenFramebuffer() {
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &textureColorbuffer);
    glDeleteRenderbuffers(1, &rbo);
    delete[] pixel_data_buffer;
}
