#include <GLFW/glfw3.h>

//global variables inputs
float output[2] = {0};

// Keyboard input callback
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    
    // Handle movement and state changes
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        switch (key) {
            case GLFW_KEY_A:
                // Move left -> decrease x position
                output[0] = -100.0f;
                break;
            case GLFW_KEY_D:
                // Move right -> increase x position
                output[0] = 100.0f;
                break;
            case GLFW_KEY_LEFT: // Testing LEFT/RIGHT for horizontal control
                 output[0] = -100.0f;
                 break;
            case GLFW_KEY_RIGHT:
                 output[0] = 100.0f;
                 break;
        }
    } else if (action == GLFW_RELEASE) {
        // Reset f_i when A or D are released
        if (key == GLFW_KEY_A || key == GLFW_KEY_D || key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT) {
            output[0] = 0.0f;
        }
    }
}