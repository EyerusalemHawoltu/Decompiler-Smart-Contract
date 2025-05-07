#ifndef UIELEMENT_H
#define UIELEMENT_H

#include "Headers.h"
#include "Shader.h"

struct UiVertex {
	glm::vec2 Position;
};

struct UiTextureVertex {
	glm::vec2 Position;
	glm::vec2 TexCoords;
};

class UiElement {
public:
	UiElement(Shader* shader, const char* texture, glm::vec2 position, glm::vec2 size, bool center = true);
	virtual void render();
	virtual void mouse_callback(GLFWwindow* window, double xpos, double ypos) = 0;
	virtual void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) = 0;
	glm::vec2 getPosition();
	glm::vec2 getCenter();
	glm::vec2 getSize();
protected:
	glm::mat4 projection = glm::ortho(0.0f, static_cast<GLfloat>(WINDOW_WIDTH), static_cast<GLfloat>(WINDOW_HEIGHT),
	                                  0.0f);
	Texture texture;
	Shader* shader;
	glm::vec2 position{};
	glm::vec2 actualPosition{};
	glm::vec2 size{};
	bool center;
	GLuint vao, vbo;
};

#endif
