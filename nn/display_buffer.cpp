// --- New variables for displaying the FBO content ---
GLuint quadVAO, quadVBO;
GLuint textureDisplayShader;
bool displayFBO = false; // The switch! Set to true to see the FBO content.

// New vertex shader for displaying the texture
const char* textureVertexShaderSource = R"glsl(
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoords;

    out vec2 TexCoords;

    void main() {
        TexCoords = aTexCoords;
        gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
    }
)glsl";

// New fragment shader for displaying the texture
const char* textureFragmentShaderSource = R"glsl(
    #version 330 core
    out vec4 FragColor;

    in vec2 TexCoords;

    uniform sampler2D screenTexture;

    void main() {
        // The texture function samples the color from the texture at the given coordinate
        FragColor = texture(screenTexture, TexCoords);
    }
)glsl";

void setupDisplayQuad() {
    // Note: We are flipping the Y texture coordinate here (0.0 becomes 1.0 and 1.0 becomes 0.0).
    // This is a common trick to automatically counteract the bottom-up way OpenGL reads textures,
    // so our final displayed image will appear right-side up, matching our drawing coordinates.
    float quadVertices[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f, // Top Left
        -1.0f, -1.0f,  0.0f, 0.0f, // Bottom Left
         1.0f, -1.0f,  1.0f, 0.0f, // Bottom Right

        -1.0f,  1.0f,  0.0f, 1.0f, // Top Left
         1.0f, -1.0f,  1.0f, 0.0f, // Bottom Right
         1.0f,  1.0f,  1.0f, 1.0f  // Top Right
    };

    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    // Position attribute
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    // Texture coordinate attribute
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    // Create and compile the new shader for displaying the texture
    GLuint vertexShader = createShader(GL_VERTEX_SHADER, textureVertexShaderSource);
    GLuint fragmentShader = createShader(GL_FRAGMENT_SHADER, textureFragmentShaderSource);
    textureDisplayShader = createProgram(vertexShader, fragmentShader);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void cleanupDisplayQuad() {
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteProgram(textureDisplayShader);
}

int main() {
    // ... (glfwInit, window creation, gladLoad) ...

    setupOffscreenFramebuffer();
    setupDisplayQuad(); // <-- Add this call

    // ... (your existing shader and geometry setup) ...

    // --- Modify key callback to toggle the display ---
    // You can add this logic to your existing keyCallback function.
    // Let's use the 'D' key to toggle the display.
    // This is just an example of how to modify your callback.
    /*
    void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_D && action == GLFW_PRESS) {
            displayFBO = !displayFBO; // Flip the boolean
            if (displayFBO) {
                std::cout << "FBO display ENABLED" << std::endl;
            } else {
                std::cout << "FBO display DISABLED" << std::endl;
            }
        }
    }
    */

    animationLoop(trackVAO, ballVAO, sledVAO, stickVAO);

    // --- Cleanup ---
    cleanupDisplayQuad(); // <-- Add this call
    cleanupOffscreenFramebuffer();
    // ... (rest of your cleanup) ...
    return 0;
}

// MODIFIED animationLoop
void animationLoop(GLuint trackVAO, GLuint ballVAO, GLuint sledVAO, GLuint stickVAO) {
    // ... (setup code before the loop) ...

    while (!glfwWindowShouldClose(window)) {
        // --- 1. RENDER TO OFF-SCREEN FBO (This part is unchanged) ---
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        // ... (physics and transform updates) ...

        drawObject(trackVAO, ..., OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT);
        drawObject(ballVAO, ..., OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT);
        // ... (draw all other objects) ...


        // --- 2. GET FRAME AND PROCESS (This part is unchanged) ---
        glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, pixel_data_buffer);
        processFrameData(pixel_data_buffer, OFFSCREEN_WIDTH, OFFSCREEN_HEIGHT);


        // --- 3. CONDITIONALLY DISPLAY FBO CONTENT ON SCREEN ---
        // Bind back to the default framebuffer (the screen)
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        if (displayFBO) {
            // Get screen dimensions for the viewport
            int screenWidth, screenHeight;
            glfwGetFramebufferSize(window, &screenWidth, &screenHeight);
            glViewport(0, 0, screenWidth, screenHeight);

            // Clear the screen's buffer
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // Use a different color to see it's working
            glClear(GL_COLOR_BUFFER_BIT);

            // Use our simple texture shader
            glUseProgram(textureDisplayShader);
            glBindVertexArray(quadVAO);

            // We don't need depth testing for a simple 2D quad overlay
            glDisable(GL_DEPTH_TEST);

            // Bind the texture that contains our rendered scene
            glBindTexture(GL_TEXTURE_2D, textureColorbuffer);

            // Draw the quad
            glDrawArrays(GL_TRIANGLES, 0, 6);
        }

        // --- 4. FINALIZE FRAME ---
        glfwSwapBuffers(window);
        glfwPollEvents();

        // ... (your sleep logic) ...
    }
}
