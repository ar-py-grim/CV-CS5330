/* 
Arpit Gandhi
March 2026

Solar System simulation in OpenGL with GLFW and GLEW.
Requires stb_image_write.h for screenshot functionality, and GLM for math operations.
Can take screenshots with 'P' key, and allows camera control with mouse and scroll wheel.
Press 'ESC' to exit.
*/

#define _CRT_SECURE_NO_WARNINGS
#define GLEW_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "stb_image_write.h"

float camYaw = 0.f, camPitch = 30.f, camDist = 60.f;
double lastX, lastY;
bool   mouseDown = false;
bool   screenshotKey = false;

// callbacks for mouse input and scroll wheel
void mouseBtnCB(GLFWwindow*, int btn, int act, int)
{
    if (btn == GLFW_MOUSE_BUTTON_LEFT) 
        mouseDown = (act == GLFW_PRESS);
}

// mouse movement callback to update camera angles when dragging
void cursorCB(GLFWwindow*, double x, double y)
{
    if (mouseDown)
    {
        camYaw += (float)(x - lastX) * 0.4f;
        camPitch += (float)(y - lastY) * 0.4f;
        camPitch = glm::clamp(camPitch, -89.f, 89.f);
    }
    lastX = x; lastY = y;
}

// scroll callback to zoom in/out by adjusting camera distance
void scrollCB(GLFWwindow*, double, double dy)
{
    camDist -= (float)dy * 2.f;
    camDist = glm::clamp(camDist, 10.f, 200.f);
}

// keyboard callback to trigger screenshot on 'P' key press
void saveScreenshot(int w, int h, const char* path)
{
    std::vector<unsigned char> pixels(w * h * 3);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    stbi_flip_vertically_on_write(1);
    stbi_write_png(path, w, h, 3, pixels.data(), w*3);
    stbi_flip_vertically_on_write(0);
    std::cout << "Screenshot saved to " << path << "\n";
}

// vertex and fragment shader sources for rendering planets
const char* vertSrc = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat3 uNormalMat;
out vec3 vWorldPos;
out vec3 vNormal;
void main()
{
    vWorldPos = vec3(uModel * vec4(aPos,1.0));
    vNormal   = normalize(uNormalMat * aNormal);
    gl_Position = uMVP * vec4(aPos,1.0);
}
)";

// fragment shader with support for emissive materials, banded coloring,
// and a special spot for Jupiter's Great Red Spot
const char* fragSrc = R"(
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
out vec4 FragColor;
uniform vec3  uColor;
uniform vec3  uLightPos;
uniform vec3  uViewPos;
uniform bool  uEmissive;
uniform bool  uBanded;
uniform float uSpinAngle;

void main()
{
    vec3 color = uColor;
    if(uBanded)
    {
        float lat = vNormal.y;
        float b1 = sin(lat * 13.0) * 0.5 + 0.5;
        vec3 dark  = vec3(0.72, 0.48, 0.30);
        vec3 light = vec3(0.96, 0.89, 0.74);
        color = mix(dark, light, b1);
        float b2 = sin(lat * 7.0 + 1.2) * 0.5 + 0.5;
        color = mix(color, vec3(0.84, 0.63, 0.42), b2 * 0.35);
        color *= 1.0 - 0.25 * (lat * lat);

        float lon = atan(vNormal.z, vNormal.x) - uSpinAngle;
        lon = mod(lon + 3.14159265, 6.28318530) - 3.14159265;
        float dLat = lat - (-0.28);
        float dLon = lon - 0.0;
        dLon = mod(dLon + 3.14159265, 6.28318530) - 3.14159265;
        float spot = dLon * dLon * 2.5 + dLat * dLat * 14.0;
        if(spot < 1.0)
        {
            vec3 grsOuter = vec3(0.85, 0.35, 0.18);
            vec3 grsInner = vec3(0.75, 0.22, 0.10);
            vec3 grs = mix(grsInner, grsOuter, spot);
            color = mix(grs, color, smoothstep(0.55, 1.0, spot));
        }
    }

    if(uEmissive)
    {
        FragColor = vec4(color, 1.0);
        return;
    }
    vec3 L = normalize(uLightPos - vWorldPos);
    vec3 V = normalize(uViewPos  - vWorldPos);
    vec3 R = reflect(-L, vNormal);
    float ambient  = 0.08;
    float diffuse  = max(dot(vNormal, L), 0.0);
    float specular = pow(max(dot(V, R), 0.0), 32.0) * 0.5;
    FragColor = vec4((ambient + diffuse + specular) * color, 1.0);
}
)";

// vertex and fragment shader sources for rendering orbit rings and planetary rings
const char* ringVertSrc = R"(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uVP;
void main()
{
 gl_Position = uVP * vec4(aPos,1.0);
}
)";

// simple fragment shader that uses a uniform color for rings
const char* ringFragSrc = R"(
#version 330 core
out vec4 FragColor;
uniform vec4 uRingColor;
void main()
{
 FragColor = uRingColor;
}
)";

// helper function to compile vertex and fragment shaders and link them into a program
GLuint compileProgram(const char* v, const char* f)
{
    auto compile = [](GLenum t, const char* s)
        {
        GLuint sh = glCreateShader(t);
        glShaderSource(sh, 1, &s, nullptr); glCompileShader(sh);
        GLint ok; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
        if (!ok)
        { 
            char l[512];
            glGetShaderInfoLog(sh, 512, nullptr, l);
            std::cerr << l;
        }
        return sh;
        };
    GLuint vs = compile(GL_VERTEX_SHADER, v), fs = compile(GL_FRAGMENT_SHADER, f);
    GLuint p = glCreateProgram();
    glAttachShader(p, vs); glAttachShader(p, fs); glLinkProgram(p);
    glDeleteShader(vs); glDeleteShader(fs);
    return p;
}

struct Mesh
{
    std::vector<float> verts;
    std::vector<unsigned> idx;
};

// generates a UV sphere mesh with given radius, stacks, and slices
Mesh makeSphere(float r, int stacks = 32, int slices = 32)
{
    Mesh m;
    for (int i = 0; i <= stacks; i++)
    {
        float phi = glm::pi<float>()*i/stacks;
        for (int j = 0; j <= slices; j++)
        {
            float theta = 2 * glm::pi<float>()*j/slices;
            float x = sin(phi)*cos(theta), y = cos(phi), z = sin(phi)*sin(theta);
            m.verts.insert(m.verts.end(), {r*x, r*y, r*z, x, y, z});
        }
    }
    for (int i = 0; i < stacks; i++)
    {
        for (int j = 0; j < slices; j++)
        {
            unsigned a = i*(slices+1)+j, b = a+slices+1;
            m.idx.insert(m.idx.end(), {a, b, a+1, b, b+1, a+1});
        }
    }
    return m;
}

// uploads a mesh to the GPU and sets up vertex attribute pointers for position and normal
GLuint uploadMesh(const Mesh& m, GLuint& vbo, GLuint& ebo)
{
    GLuint vao;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, m.verts.size() * 4, m.verts.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &ebo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m.idx.size() * 4, m.idx.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, (void*)0);  glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, (void*)12); glEnableVertexAttribArray(1);
    return vao;
}

// generates a VAO for a circular orbit ring with given radius and number of segments
GLuint makeOrbitVAO(float radius, GLuint& vbo, int segs = 128)
{
    std::vector<float> pts;
    for (int i = 0; i <= segs; i++)
    {
        float a = 2*glm::pi<float>()*i/segs;
        pts.push_back(radius*cos(a)); pts.push_back(0.f); pts.push_back(radius*sin(a));
    }
    GLuint vao;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, pts.size() * 4, pts.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, (void*)0); glEnableVertexAttribArray(0);
    return vao;
}

// generates a VAO for a planet ring (like Saturn's) with given inner and outer radius, and number of segments
GLuint makePlanetRingVAO(float innerR, float outerR, GLuint& vbo, GLuint& ebo,
    int& idxCount, int segs = 128)
{
    std::vector<float>    pts;
    std::vector<unsigned> idx;
    for (int i = 0; i <= segs; i++)
    {
        float a = 2 * glm::pi<float>()*i/segs;
        float c = cos(a), s = sin(a);
        pts.insert(pts.end(), {innerR*c, 0.f, innerR*s});
        pts.insert(pts.end(), {outerR*c, 0.f, outerR*s});
    }
    for (int i = 0; i < segs; i++)
    {
        unsigned b = i*2;
        idx.insert(idx.end(), {b, b+1, b+2, b+1, b+3, b+2});
    }

    idxCount = (int)idx.size();
    GLuint vao;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, pts.size() * 4, pts.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &ebo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * 4, idx.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, (void*)0); glEnableVertexAttribArray(0);
    return vao;
}

struct Planet
{
    const char* name;
    float  radius, orbit, speed, spinSpeed;
    glm::vec3 color;
    float  moonOrbit, moonRadius, moonSpeed;
};

const Planet planets[] = {
    {"Mercury",  0.6f,  8.f,  1.60f, 2.0f, {0.76f,0.70f,0.66f}, 0.f,  0.f,  0.f},
    {"Venus",    1.0f, 13.f,  1.17f, 0.5f, {0.90f,0.75f,0.45f}, 0.f,  0.f,  0.f},
    {"Earth",    1.1f, 19.f,  1.00f, 1.5f, {0.25f,0.55f,0.90f}, 2.5f, 0.3f, 3.0f},
    {"Mars",     0.8f, 26.f,  0.80f, 1.4f, {0.80f,0.35f,0.20f}, 0.f,  0.f,  0.f},
    {"Jupiter",  2.8f, 38.f,  0.43f, 3.0f, {0.85f,0.72f,0.55f}, 0.f,  0.f,  0.f},
    {"Saturn",   2.3f, 52.f,  0.32f, 2.8f, {0.95f,0.85f,0.65f}, 0.f,  0.f,  0.f},
    {"Uranus",   1.7f, 64.f,  0.23f, 1.8f, {0.60f,0.85f,0.90f}, 0.f,  0.f,  0.f},
    {"Neptune",  1.6f, 76.f,  0.18f, 1.7f, {0.25f,0.40f,0.85f}, 0.f,  0.f,  0.f},
};

const int NUM_PLANETS = 8;
const int SATURN_IDX = 5;
const int JUPITER_IDX = 4;

int main()
{
    if (!glfwInit())
    {
        return -1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

	// set window hints based on primary monitor's video mode for fullscreen
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    glfwWindowHint(GLFW_RED_BITS, mode->redBits);
    glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
    glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
    glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

    GLFWwindow* win = glfwCreateWindow(mode->width, mode->height, "Solar System", monitor, nullptr);

	// check if window creation succeeded
    glfwMakeContextCurrent(win);
	// set up input callbacks for mouse and scroll events
    glfwSetMouseButtonCallback(win, mouseBtnCB);
    glfwSetCursorPosCallback(win, cursorCB);
    glfwSetScrollCallback(win, scrollCB);

    glewExperimental = GL_TRUE; glewInit();
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    GLuint prog = compileProgram(vertSrc, fragSrc);
    GLuint ringProg = compileProgram(ringVertSrc, ringFragSrc);

    GLuint sunVBO, sunEBO;
    Mesh   sunMesh = makeSphere(4.0f);
    GLuint sunVAO = uploadMesh(sunMesh, sunVBO, sunEBO);

    GLuint pVAO[NUM_PLANETS], pVBO[NUM_PLANETS], pEBO[NUM_PLANETS];
    GLuint mVAO[NUM_PLANETS], mVBO[NUM_PLANETS], mEBO[NUM_PLANETS];
    GLuint orbitVAO[NUM_PLANETS], orbitVBO[NUM_PLANETS];
    GLuint moonOrbitVAO[NUM_PLANETS], moonOrbitVBO[NUM_PLANETS];

    GLuint sRingVAO, sRingVBO, sRingEBO;
    int    sRingIdxCount = 0;
    sRingVAO = makePlanetRingVAO(3.2f, 5.8f, sRingVBO, sRingEBO, sRingIdxCount);

	// generate VAOs and VBOs for each planet, their orbits, and moons if applicable
    for (int i = 0; i < NUM_PLANETS; i++)
    {
        Mesh pm = makeSphere(planets[i].radius);
        pVAO[i] = uploadMesh(pm, pVBO[i], pEBO[i]);
        orbitVAO[i] = makeOrbitVAO(planets[i].orbit, orbitVBO[i]);
        if (planets[i].moonOrbit > 0)
        {
            Mesh mm = makeSphere(planets[i].moonRadius);
            mVAO[i] = uploadMesh(mm, mVBO[i], mEBO[i]);
            moonOrbitVAO[i] = makeOrbitVAO(planets[i].moonOrbit, moonOrbitVBO[i]);
        }
    }

    float orbitAngles[NUM_PLANETS] = {};
    float spinAngles[NUM_PLANETS] = {};
    float moonAngles[NUM_PLANETS] = {};
    int   shotCount = 0;

    double prevTime = glfwGetTime();

	// Main render loop
    while (!glfwWindowShouldClose(win))
    {
        double now = glfwGetTime();
        float  dt = (float)(now - prevTime);
        prevTime = now;

        for (int i = 0; i < NUM_PLANETS; i++)
        {
            orbitAngles[i]+= planets[i].speed*dt;
            spinAngles[i]+= planets[i].spinSpeed*dt;
            moonAngles[i]+= planets[i].moonSpeed*dt;
        }

        int fbW, fbH;
        glfwGetFramebufferSize(win, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);

        glClearColor(0.05f, 0.05f, 0.15f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        float yawR = glm::radians(camYaw);
        float pitchR = glm::radians(camPitch);
		// calculate camera position in Cartesian coordinates based on spherical angles and distance
        glm::vec3 eye(
            camDist * cos(pitchR)*sin(yawR),
            camDist * sin(pitchR),
            camDist * cos(pitchR)*cos(yawR)
        );
        glm::mat4 view = glm::lookAt(eye, glm::vec3(0), glm::vec3(0, 1, 0));
        glm::mat4 proj = glm::perspective(glm::radians(45.f), (float)fbW / fbH, 0.1f, 500.f);
        glm::mat4 VP = proj*view;

        // Orbit rings
        glUseProgram(ringProg);
        glUniformMatrix4fv(glGetUniformLocation(ringProg, "uVP"), 1, GL_FALSE, glm::value_ptr(VP));
        glUniform4f(glGetUniformLocation(ringProg, "uRingColor"), 1.f, 1.f, 1.f, 0.18f);
        for (int i = 0; i < NUM_PLANETS; i++)
        {
            glBindVertexArray(orbitVAO[i]);
            glDrawArrays(GL_LINE_STRIP, 0, 129);
        }

        // Sun
        glUseProgram(prog);
        glUniform3fv(glGetUniformLocation(prog, "uLightPos"), 1, glm::value_ptr(glm::vec3(0)));
        glUniform3fv(glGetUniformLocation(prog, "uViewPos"), 1, glm::value_ptr(eye));
        glUniform1i(glGetUniformLocation(prog, "uEmissive"), 1);
        glUniform1i(glGetUniformLocation(prog, "uBanded"), 0);
        glUniform3f(glGetUniformLocation(prog, "uColor"), 1.0f, 0.9f, 0.3f);
        {
            glm::mat4 model(1.f);
            glm::mat4 mvp = VP * model;
            glm::mat3 nm = glm::transpose(glm::inverse(glm::mat3(model)));
            glUniformMatrix4fv(glGetUniformLocation(prog, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
            glUniformMatrix4fv(glGetUniformLocation(prog, "uModel"), 1, GL_FALSE, glm::value_ptr(model));
            glUniformMatrix3fv(glGetUniformLocation(prog, "uNormalMat"), 1, GL_FALSE, glm::value_ptr(nm));
        }
        glBindVertexArray(sunVAO);
        glDrawElements(GL_TRIANGLES, (GLsizei)sunMesh.idx.size(), GL_UNSIGNED_INT, 0);

        glUniform1i(glGetUniformLocation(prog, "uEmissive"), 0);

		// Planets and moons
        for (int i = 0; i < NUM_PLANETS; i++)
        {
            float oa = orbitAngles[i];
            glm::vec3 pPos(planets[i].orbit * cos(oa), 0.f, planets[i].orbit * sin(oa));

            glm::mat4 model = glm::translate(glm::mat4(1.f), pPos);
            if (i == SATURN_IDX)
            {
                model = glm::rotate(model, glm::radians(27.f), glm::vec3(0.f, 0.f, 1.f));
            }
            model = glm::rotate(model, spinAngles[i], glm::vec3(0, 1, 0));

            glm::mat4 mvp = VP * model;
            glm::mat3 nm = glm::transpose(glm::inverse(glm::mat3(model)));

            if (i == JUPITER_IDX)
            {
                glUniform1i(glGetUniformLocation(prog, "uBanded"), 1);
                glUniform1f(glGetUniformLocation(prog, "uSpinAngle"), spinAngles[i]);
            }
            else
            {
                glUniform1i(glGetUniformLocation(prog, "uBanded"), 0);
            }

            glUniform3fv(glGetUniformLocation(prog, "uColor"), 1, glm::value_ptr(planets[i].color));
            glUniformMatrix4fv(glGetUniformLocation(prog, "uMVP"), 1, GL_FALSE, glm::value_ptr(mvp));
            glUniformMatrix4fv(glGetUniformLocation(prog, "uModel"), 1, GL_FALSE, glm::value_ptr(model));
            glUniformMatrix3fv(glGetUniformLocation(prog, "uNormalMat"), 1, GL_FALSE, glm::value_ptr(nm));

            glBindVertexArray(pVAO[i]);
            glDrawElements(GL_TRIANGLES, (GLsizei)32 * 32 * 6, GL_UNSIGNED_INT, 0);

            if (i == SATURN_IDX)
            {
                glUseProgram(ringProg);
                glm::mat4 ringModel = glm::translate(glm::mat4(1.f), pPos);
                ringModel = glm::rotate(ringModel, glm::radians(27.f), glm::vec3(0.f, 0.f, 1.f));
                glm::mat4 ringVP = VP * ringModel;
                glUniformMatrix4fv(glGetUniformLocation(ringProg, "uVP"), 1, GL_FALSE, glm::value_ptr(ringVP));
                glUniform4f(glGetUniformLocation(ringProg, "uRingColor"), 0.88f, 0.78f, 0.55f, 0.55f);
                glBindVertexArray(sRingVAO);
                glDrawElements(GL_TRIANGLES, sRingIdxCount, GL_UNSIGNED_INT, 0);
                glUseProgram(prog);
                glUniform3fv(glGetUniformLocation(prog, "uLightPos"), 1, glm::value_ptr(glm::vec3(0)));
                glUniform3fv(glGetUniformLocation(prog, "uViewPos"), 1, glm::value_ptr(eye));
                glUniform1i(glGetUniformLocation(prog, "uEmissive"), 0);
                glUniform1i(glGetUniformLocation(prog, "uBanded"), 0);
            }

            if (planets[i].moonOrbit > 0)
            {
                glUseProgram(ringProg);
                glm::mat4 ringVP = VP * glm::translate(glm::mat4(1.f), pPos);
                glUniformMatrix4fv(glGetUniformLocation(ringProg, "uVP"), 1, GL_FALSE, glm::value_ptr(ringVP));
                glUniform4f(glGetUniformLocation(ringProg, "uRingColor"), 1.f, 1.f, 1.f, 0.18f);
                glBindVertexArray(moonOrbitVAO[i]);
                glDrawArrays(GL_LINE_STRIP, 0, 129);

                glUseProgram(prog);
                float ma = moonAngles[i];
                glm::vec3 mPos = pPos + glm::vec3(
                    planets[i].moonOrbit * cos(ma), 0.f,
                    planets[i].moonOrbit * sin(ma));

                glm::mat4 mModel = glm::translate(glm::mat4(1.f), mPos);
                glm::mat4 mMvp = VP * mModel;
                glm::mat3 mNm = glm::transpose(glm::inverse(glm::mat3(mModel)));

                glUniform1i(glGetUniformLocation(prog, "uBanded"), 0);
                glUniform3f(glGetUniformLocation(prog, "uColor"), 0.8f, 0.8f, 0.8f);
                glUniformMatrix4fv(glGetUniformLocation(prog, "uMVP"), 1, GL_FALSE, glm::value_ptr(mMvp));
                glUniformMatrix4fv(glGetUniformLocation(prog, "uModel"), 1, GL_FALSE, glm::value_ptr(mModel));
                glUniformMatrix3fv(glGetUniformLocation(prog, "uNormalMat"), 1, GL_FALSE, glm::value_ptr(mNm));

                glBindVertexArray(mVAO[i]);
                glDrawElements(GL_TRIANGLES, (GLsizei)32 * 32 * 6, GL_UNSIGNED_INT, 0);
            }
        }

        glfwSwapBuffers(win);
        glfwPollEvents();

		// check for 'ESC' key press to close the window
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(win, GLFW_TRUE);
        }

        bool pDown = glfwGetKey(win, GLFW_KEY_P) == GLFW_PRESS;
        if (pDown && !screenshotKey)
        {
            char path[64];
            std::snprintf(path, sizeof(path), "screenshot_%03d.png", ++shotCount);
            saveScreenshot(fbW, fbH, path);
        }
        screenshotKey = pDown;
    }

	// cleanup
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}