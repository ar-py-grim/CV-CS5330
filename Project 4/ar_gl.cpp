/*
Arpit Gandhi
March 2026

Code to use OpenCV's ChArUco board detection and OpenGL for rendering solar system on top of the detected board.
*/

#define GLEW_STATIC
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// CharucoBoard config
static const int   SQ_X = 10;
static const int   SQ_Y = 7;
static const float SQ_SZ = 1.0f;
static const float MK_SZ = 0.6f;

// Shaders
// simple textured quad shader for background video feed
static const char* BG_VERT = R"(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 vUV;
void main()
{ 
 vUV = aUV;
 gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

// simple shader to render the camera feed as background using a textured quad
static const char* BG_FRAG = R"(
#version 330 core
in vec2 vUV; out vec4 F;
uniform sampler2D uTex;
void main()
{
 F = texture(uTex, vUV);
}
)";

// shader for rendering planets with optional banding and a spot on Jupiter
static const char* OBJ_VERT = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
uniform mat4 uMVP;
uniform mat4 uModel;
uniform mat3 uNormMat;
out vec3 wPos; out vec3 wNorm;
void main()
{
    wPos  = vec3(uModel * vec4(aPos, 1.0));
    wNorm = normalize(uNormMat * aNorm);
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)";

// fragment shader that implements Phong lighting and adds banding and a red spot for Jupiter based on latitude and longitude
static const char* OBJ_FRAG = R"(
#version 330 core
in vec3 wPos; in vec3 wNorm; out vec4 F;
uniform vec3  uLight;
uniform vec3  uView;
uniform vec3  uColor;
uniform float uEmit;
uniform bool  uBanded;
uniform float uSpinAngle;

void main()
{
    vec3 color = uColor;
    if (uBanded)
    {
        float lat = wNorm.y;
        float b1  = sin(lat * 13.0) * 0.5 + 0.5;
        vec3 dark  = vec3(0.72, 0.48, 0.30);
        vec3 light = vec3(0.96, 0.89, 0.74);
        color = mix(dark, light, b1);
        float b2 = sin(lat * 7.0 + 1.2) * 0.5 + 0.5;
        color = mix(color, vec3(0.84, 0.63, 0.42), b2 * 0.35);
        color *= 1.0 - 0.25 * (lat * lat);
        float lon  = atan(wNorm.z, wNorm.x) - uSpinAngle;
        lon  = mod(lon  + 3.14159265, 6.28318530) - 3.14159265;
        float dLat = lat - (-0.28);
        float dLon = lon;
        dLon = mod(dLon + 3.14159265, 6.28318530) - 3.14159265;
        float spot = dLon * dLon * 2.5 + dLat * dLat * 14.0;
        if (spot < 1.0)
        {
            vec3 grsO = vec3(0.85, 0.35, 0.18);
            vec3 grsI = vec3(0.75, 0.22, 0.10);
            vec3 grs  = mix(grsI, grsO, spot);
            color = mix(grs, color, smoothstep(0.55, 1.0, spot));
        }
    }
    vec3 l = normalize(uLight - wPos);
    vec3 v = normalize(uView  - wPos);
    vec3 r = reflect(-l, wNorm);
    float diff = max(dot(wNorm, l), 0.0);
    float spec = pow(max(dot(v, r), 0.0), 48.0);
    float lit  = mix(0.08 + diff, 1.0, uEmit);
    F = vec4(lit * color + spec * (1.0 - uEmit), 1.0);
}
)";

// simple shader for rendering rings as flat colored discs
static const char* RING_VERT = R"(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
)";

// simple fragment shader that uses a uniform color for rings
static const char* RING_FRAG = R"(
#version 330 core
out vec4 F;
uniform vec4 uColor;
void main() { F = uColor; }
)";

// utility function to compile shaders and link them into a program
static GLuint ar_makeProgram(const char* vs, const char* fs)
{
    auto compile = [](GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr); glCompileShader(s);
        GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) { char b[512]; glGetShaderInfoLog(s, 512, nullptr, b); cerr << b << endl; }
        return s;
        };
    GLuint v = compile(GL_VERTEX_SHADER, vs);
    GLuint f = compile(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

// utility function to convert OpenCV camera intrinsics to an OpenGL projection matrix
static glm::mat4 ar_cvK2proj(const Mat& K, int w, int h, float zn = 0.01f, float zf = 200.f)
{
    float fx = K.at<double>(0, 0), fy = K.at<double>(1, 1);
    float cx = K.at<double>(0, 2), cy = K.at<double>(1, 2);
    glm::mat4 P(0.f);
    P[0][0] = 2.f*fx/w;
    P[1][1] = 2.f*fy/h;
    P[2][0] = 1.f-2.f*cx/w;
    P[2][1] = 2.f*cy/h-1.f;
    P[2][2] = -(zf+zn)/(zf-zn);
    P[2][3] = -1.f;
    P[3][2] = -2.f*zf*zn/(zf-zn);
    return P;
}

// utility function to convert OpenCV's rotation and translation vectors to an OpenGL model-view matrix
static glm::mat4 ar_cvRT2mv(const Mat& rvec, const Mat& tvec)
{
    Mat R; Rodrigues(rvec, R);
    glm::mat4 MV(1.f);
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            MV[c][r] = (float)R.at<double>(r, c);
    MV[3][0] = (float)tvec.at<double>(0);
    MV[3][1] = (float)tvec.at<double>(1);
    MV[3][2] = (float)tvec.at<double>(2);
    glm::mat4 flip(1.f);
    flip[1][1] = -1.f; flip[2][2] = -1.f;
    return flip * MV;
}

struct ArMesh 
{ 
    vector<float> v;
    vector<unsigned> i;
};

// generates a UV sphere mesh with given radius, stacks, and slices
static ArMesh ar_uvSphere(float r = 1.f, int stk = 32, int sl = 32)
{
    ArMesh m;
    for (int i = 0; i <= stk; i++)
    {
        float phi = (float)CV_PI*i/stk;
        for (int j = 0; j <= sl; j++)
        {
            float th = 2.f*(float)CV_PI*j/sl;
            float x = sinf(phi) * cosf(th), y = cosf(phi), z = sinf(phi)*sinf(th);
            m.v.insert(m.v.end(), {r*x, r*y, r*z, x, y, z});
        }
    }

    for (int i = 0; i < stk; i++)
    {
        for (int j = 0; j < sl; j++)
        {
            unsigned a = i*(sl+1)+j, b = a+sl+1;
            m.i.insert(m.i.end(), {a, b, a+1, b, b+1, a+1});
        }
    }
    return m;
}

// uploads a mesh to the GPU and sets up vertex attribute pointers for position and normal
static GLuint ar_uploadMesh(const ArMesh& m, GLuint& vbo, GLuint& ebo)
{
    GLuint vao;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, m.v.size() * sizeof(float), m.v.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &ebo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m.i.size() * sizeof(unsigned), m.i.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    return vao;
}

// generates a VAO for a planet ring (like Saturn's) with given inner and outer radius, and number of segments
static GLuint ar_makeSaturnRingVAO(float innerR, float outerR,
    GLuint& vbo, GLuint& ebo,
    int& idxCount, int segs = 128)
{
    vector<float>    pts;
    vector<unsigned> idx;
    for (int i = 0; i <= segs; i++)
    {
        float a = 2.f*(float)CV_PI*i/segs;
        float c = cosf(a), s = sinf(a);
        // XZ plane ring lies flat around Saturn's equator
        pts.insert(pts.end(), {innerR*c, 0.f, innerR*s});
        pts.insert(pts.end(), {outerR*c, 0.f, outerR*s});
    }

    for (int i = 0; i < segs; i++)
    {
        unsigned b = i * 2;
        idx.insert(idx.end(), {b, b+1, b+2, b+1, b+3, b+2});
    }

    idxCount = (int)idx.size();
    GLuint vao;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, pts.size() * sizeof(float), pts.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &ebo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(unsigned), idx.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    return vao;
}

// utility function to convert an OpenCV Mat image to an OpenGL texture, with RGB conversion and vertical flip
static GLuint ar_matToTexture(const Mat& img)
{
    Mat rgb; cvtColor(img, rgb, COLOR_BGR2RGB); flip(rgb, rgb, 0);
    GLuint tex;
    glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rgb.cols, rgb.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
    return tex;
}

// Planet data
struct ArPlanet
{
    const char* name;
    float       orbit, speed, radius, spinSpeed;
    glm::vec3   color;
    bool        isSun, isBanded;
    float       moonOrbit, moonRadius, moonSpeed;
};

static const int JUPITER_IDX = 5;
static const int SATURN_IDX = 6;
static const int NUM_PLANETS = 9;

// data for the Sun and 8 planets, with their orbital radius, speed, size, color, and moon parameters
static const ArPlanet AR_PLANETS[NUM_PLANETS] =
{
    {"Sun",        0.0f,  0.00f, 0.45f,  0.30f, {1.00f,0.85f,0.00f},  true,  false,  0.0f,  0.00f,  0.0f},
    {"Mercury",    0.6f,  2.20f, 0.10f,  2.00f, {0.76f,0.70f,0.66f},  false, false,  0.0f,  0.00f,  0.0f},
    {"Venus",      0.9f,  1.40f, 0.17f,  0.50f, {0.90f,0.75f,0.45f},  false, false,  0.0f,  0.00f,  0.0f},
    {"Earth",      1.3f,  0.90f, 0.19f,  1.50f, {0.25f,0.55f,0.90f},  false, false,  0.38f, 0.07f,  3.0f},
    {"Mars",       1.8f,  0.55f, 0.13f,  1.40f, {0.80f,0.35f,0.20f},  false, false,  0.0f,  0.00f,  0.0f},
    {"Jupiter",    2.5f,  0.35f, 0.30f,  3.00f, {0.85f,0.72f,0.55f},  false, true,   0.0f,  0.00f,  0.0f},
    {"Saturn",     3.2f,  0.25f, 0.24f,  2.80f, {0.95f,0.85f,0.65f},  false, false,  0.0f,  0.00f,  0.0f},
    {"Uranus",     3.9f,  0.18f, 0.18f,  1.80f, {0.60f,0.85f,0.90f},  false, false,  0.0f,  0.00f,  0.0f},
    {"Neptune",    4.5f,  0.14f, 0.17f,  1.70f, {0.25f,0.40f,0.85f},  false, false,  0.0f,  0.00f,  0.0f},
};

// main function that sets up OpenCV, OpenGL, and the rendering loop to display the solar system 
// on top of the detected Charuco board
int arGL()
{
    const string spacePath = "C:/Users/ASUS/Desktop/CS5330/Project 4/space.png";

    Mat spaceImg = imread(spacePath);
    if (spaceImg.empty())
    { 
        cerr << "Cannot open space image\n";
        return -1;
    }

    Mat cameraMatrix, distCoeffs;
    FileStorage fsc("config/calibration.yaml", FileStorage::READ);
    if (!fsc.isOpened())
    { 
        cerr << "Cannot open calibration.yaml\n";
        return -1;
    }
    fsc["camera_matrix"] >> cameraMatrix;
    fsc["distortion_coeffs"] >> distCoeffs;
    fsc.release();

    VideoCapture cap(0);
    if (!cap.isOpened())
    { 
        cerr << "Cannot open camera\n";
        return -1;
    }

	// setting camera properties
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);
    Mat probe; cap >> probe;
    int camW = probe.cols, camH = probe.rows;
    cout << "Camera: " << camW << "x" << camH << "\n";

    if (!glfwInit())
    {
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

	// Create a fullscreen window using the primary monitor's video mode dimensions
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    int WIN_W = mode->width, WIN_H = mode->height;

    resize(spaceImg, spaceImg, Size(WIN_W, WIN_H));

	// Create a fullscreen window
    GLFWwindow* win = glfwCreateWindow(WIN_W, WIN_H, "AR Solar System", nullptr, nullptr);
    glfwMakeContextCurrent(win);

    int fbW, fbH;
    glfwGetFramebufferSize(win, &fbW, &fbH);

    glewExperimental = GL_TRUE; glewInit();
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glViewport(0, 0, fbW, fbH);

    // Background quad
    float bgV[] = {-1,-1,0,1,  1,-1,1,1,  1,1,1,0,
                    -1,-1,0,1,  1, 1,1,0,  -1,1,0,0};
    GLuint bgVAO, bgVBO;
    glGenVertexArrays(1, &bgVAO); glBindVertexArray(bgVAO);
    glGenBuffers(1, &bgVBO); glBindBuffer(GL_ARRAY_BUFFER, bgVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(bgV), bgV, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    GLuint spaceTex = ar_matToTexture(spaceImg);

    // Sphere mesh (shared by all bodies)
    ArMesh sph = ar_uvSphere(1.0f, 32, 32);
    GLuint sphVBO, sphEBO;
    GLuint sphVAO = ar_uploadMesh(sph, sphVBO, sphEBO);

    // Dynamic orbit line-strip (reused per planet per frame)
    const int RING_SEGS = 128;
    GLuint ringVAO, ringVBO;
    glGenVertexArrays(1, &ringVAO); glBindVertexArray(ringVAO);
    glGenBuffers(1, &ringVBO); glBindBuffer(GL_ARRAY_BUFFER, ringVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3*(RING_SEGS+1), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

	// Saturn's ring mesh
    GLuint satRingVAO, satRingVBO, satRingEBO;
    int    satRingIdxCount = 0;
    satRingVAO = ar_makeSaturnRingVAO(0.34f, 0.61f,
        satRingVBO, satRingEBO, satRingIdxCount);

    GLuint bgProg = ar_makeProgram(BG_VERT, BG_FRAG);
    GLuint objProg = ar_makeProgram(OBJ_VERT, OBJ_FRAG);
    GLuint ringProg = ar_makeProgram(RING_VERT, RING_FRAG);

    aruco::Dictionary      dict = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    aruco::CharucoBoard    board(Size(SQ_X, SQ_Y), SQ_SZ, MK_SZ, dict);
    aruco::CharucoDetector detector(board);

	// adjust the projection matrix to account for aspect ratio differences between the camera feed and the window
    float camAspect = (float)camW/camH;
    float winAspect = (float)WIN_W/WIN_H;
    glm::mat4 proj = ar_cvK2proj(cameraMatrix, camW, camH);
    proj[0][0]*= camAspect/winAspect;

    const glm::vec3 SYS_CENTRE(4.5f, -3.0f, 0.5f);
    const glm::vec3 LIGHT_POS(4.5f, -3.0f, 12.0f);

    float spinAngles[NUM_PLANETS] = {};
    float moonAngles[NUM_PLANETS] = {};
    double prevT = (double)getTickCount()/getTickFrequency();

    cout << "Point camera at ChArUco board  |  Q = quit\n";

    Mat frame;
    while (!glfwWindowShouldClose(win))
    {
        cap >> frame;
        if (frame.empty()) break;

        double now = (double)getTickCount()/getTickFrequency();
        float  dt = (float)(now-prevT);
        prevT = now;
		// update planet and moon angles speeds
        for (int i = 0; i < NUM_PLANETS; i++)
        {
            spinAngles[i] += AR_PLANETS[i].spinSpeed*dt;
            moonAngles[i] += AR_PLANETS[i].moonSpeed*dt;
        }

        glfwGetFramebufferSize(win, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Background
        glDepthMask(GL_FALSE);
        glUseProgram(bgProg);
        glBindVertexArray(bgVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, spaceTex);
        glUniform1i(glGetUniformLocation(bgProg, "uTex"), 0);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glDepthMask(GL_TRUE);

        // Board detection
        vector<int>             markerIds;
        vector<vector<Point2f>> markerCorners;
        vector<int>             charucoIds;
        vector<Point2f>         charucoCorners;
        detector.detectBoard(frame, charucoCorners, charucoIds,
            markerCorners, markerIds);

        if ((int)charucoIds.size() >= 4)
        {
            vector<Point3f> op; vector<Point2f> ip;
            board.matchImagePoints(charucoCorners, charucoIds, op, ip);

            if ((int)op.size() >= 6)
            {
                Mat rvec, tvec;
                try
                {
                    if (solvePnP(op, ip, cameraMatrix, distCoeffs, rvec, tvec))
                    {
                        glm::mat4 mv = ar_cvRT2mv(rvec, tvec);
                        double t = (double)getTickCount()/getTickFrequency();

						// Render planets
                        for (int pi = 0; pi < NUM_PLANETS; pi++)
                        {
                            const ArPlanet& pl = AR_PLANETS[pi];
                            float orbitAngle = (float)(pl.speed*t);

                            glm::vec3 pos = SYS_CENTRE + glm::vec3(pl.orbit*cosf(orbitAngle),
                                    pl.orbit*sinf(orbitAngle), 0.f);

                            // Orbit ring
                            if (pl.orbit > 0.f)
                            {
                                vector<float> rpts;
                                rpts.reserve(3*(RING_SEGS+1));
                                for (int k = 0; k <= RING_SEGS; k++)
                                {
                                    float a = 2.f*(float)CV_PI*k/RING_SEGS;
                                    rpts.push_back(SYS_CENTRE.x + pl.orbit*cosf(a));
                                    rpts.push_back(SYS_CENTRE.y + pl.orbit*sinf(a));
                                    rpts.push_back(SYS_CENTRE.z);
                                }
                                glBindVertexArray(ringVAO);
                                glBindBuffer(GL_ARRAY_BUFFER, ringVBO);
                                glBufferSubData(GL_ARRAY_BUFFER, 0, rpts.size()*sizeof(float), rpts.data());

                                glUseProgram(ringProg);
                                glm::mat4 ringMVP = proj*mv;
                                glUniformMatrix4fv(glGetUniformLocation(ringProg, "uMVP"), 1,
                                         GL_FALSE, glm::value_ptr(ringMVP));
                                glUniform4f(glGetUniformLocation(ringProg, "uColor"), 0.4f, 0.4f, 0.6f, 0.55f);
                                glDrawArrays(GL_LINE_STRIP, 0, RING_SEGS+1);
                            }

                            // Planet model matrix
                            glm::mat4 model = mv*glm::translate(glm::mat4(1.f), pos);

                            if (pi == SATURN_IDX)
                            {
                                model = model*glm::rotate(glm::mat4(1.f), glm::radians(27.f),
                                    glm::vec3(1.f, 0.f, 0.f));
                            }
                            model = model*glm::rotate(glm::mat4(1.f), spinAngles[pi],
                                    glm::vec3(0.f, 1.f, 0.f))*glm::scale(glm::mat4(1.f), glm::vec3(pl.radius));

                            glm::mat4 mvp = proj*model;
                            glm::mat3 normMat = glm::transpose(glm::inverse(glm::mat3(model)));

                            glUseProgram(objProg);
                            glUniformMatrix4fv(glGetUniformLocation(objProg, "uMVP"), 1, 
                                                          GL_FALSE, glm::value_ptr(mvp));
                            glUniformMatrix4fv(glGetUniformLocation(objProg, "uModel"), 1, 
                                                          GL_FALSE, glm::value_ptr(model));
                            glUniformMatrix3fv(glGetUniformLocation(objProg, "uNormMat"), 1,
                                                          GL_FALSE, glm::value_ptr(normMat));
                            glUniform3fv(glGetUniformLocation(objProg, "uColor"), 1, glm::value_ptr(pl.color));
                            glUniform3f(glGetUniformLocation(objProg, "uLight"), LIGHT_POS.x, LIGHT_POS.y, LIGHT_POS.z);
                            glUniform3f(glGetUniformLocation(objProg, "uView"), 0.f, 0.f, 0.f);
                            glUniform1f(glGetUniformLocation(objProg, "uEmit"), pl.isSun ? 1.f : 0.f);
                            glUniform1i(glGetUniformLocation(objProg, "uBanded"), pi == JUPITER_IDX ? 1 : 0);
                            glUniform1f(glGetUniformLocation(objProg, "uSpinAngle"), spinAngles[pi]);

                            glBindVertexArray(sphVAO);
                            glDrawElements(GL_TRIANGLES, (GLsizei)sph.i.size(), GL_UNSIGNED_INT, 0);

                            // Saturn planetary ring
                            if (pi == SATURN_IDX)
                            {
                                glm::mat4 rModel = mv*glm::translate(glm::mat4(1.f), pos)
                                    * glm::rotate(glm::mat4(1.f), glm::radians(27.f), glm::vec3(1.f, 0.f, 0.f));
                                glm::mat4 rMVP = proj*rModel;

                                glUseProgram(ringProg);
                                glUniformMatrix4fv(glGetUniformLocation(ringProg, "uMVP"), 1, 
                                                        GL_FALSE, glm::value_ptr(rMVP));
                                glUniform4f(glGetUniformLocation(ringProg, "uColor"), 0.88f, 0.78f, 0.55f, 0.70f);
                                glBindVertexArray(satRingVAO);
                                glDrawElements(GL_TRIANGLES, satRingIdxCount, GL_UNSIGNED_INT, 0);
                            }

                            // Earth's Moon
                            if (pl.moonOrbit > 0.f)
                            {
                                // Moon orbit ring
                                vector<float> mpts;
                                mpts.reserve(3*(RING_SEGS+1));
                                for (int k = 0; k <= RING_SEGS; k++)
                                {
                                    float a = 2.f*(float)CV_PI*k/RING_SEGS;
                                    mpts.push_back(pos.x + pl.moonOrbit*cosf(a));
                                    mpts.push_back(pos.y + pl.moonOrbit*sinf(a));
                                    mpts.push_back(pos.z);
                                }
                                glBindVertexArray(ringVAO);
                                glBindBuffer(GL_ARRAY_BUFFER, ringVBO);
                                glBufferSubData(GL_ARRAY_BUFFER, 0, mpts.size()*sizeof(float), mpts.data());
                                glUseProgram(ringProg);
                                glm::mat4 mRingMVP = proj*mv;
                                glUniformMatrix4fv(glGetUniformLocation(ringProg, "uMVP"), 1, 
                                                          GL_FALSE, glm::value_ptr(mRingMVP));
                                glUniform4f(glGetUniformLocation(ringProg, "uColor"), 1.f, 1.f, 1.f, 0.25f);
                                glDrawArrays(GL_LINE_STRIP, 0, RING_SEGS+1);

                                // Moon sphere
                                glm::vec3 mPos = pos + glm::vec3(pl.moonOrbit*cosf(moonAngles[pi]),
                                    pl.moonOrbit*sinf(moonAngles[pi]), 0.f);
                                glm::mat4 mModel = mv*glm::translate(glm::mat4(1.f), mPos)*glm::scale(glm::mat4(1.f),
                                                                     glm::vec3(pl.moonRadius));
                                glm::mat4 mMVP = proj*mModel;
                                glm::mat3 mNormMat = glm::transpose(glm::inverse(glm::mat3(mModel)));

                                glUseProgram(objProg);
                                glUniformMatrix4fv(glGetUniformLocation(objProg, "uMVP"), 1, 
                                       GL_FALSE, glm::value_ptr(mMVP));
                                glUniformMatrix4fv(glGetUniformLocation(objProg, "uModel"), 1,
                                       GL_FALSE, glm::value_ptr(mModel));
                                glUniformMatrix3fv(glGetUniformLocation(objProg, "uNormMat"), 1,
                                       GL_FALSE, glm::value_ptr(mNormMat));
                                glUniform3f(glGetUniformLocation(objProg, "uColor"), 0.80f, 0.80f, 0.80f);
                                glUniform3f(glGetUniformLocation(objProg, "uLight"), LIGHT_POS.x, LIGHT_POS.y, LIGHT_POS.z);
                                glUniform3f(glGetUniformLocation(objProg, "uView"), 0.f, 0.f, 0.f);
                                glUniform1f(glGetUniformLocation(objProg, "uEmit"), 0.f);
                                glUniform1i(glGetUniformLocation(objProg, "uBanded"), 0);
                                glBindVertexArray(sphVAO);
                                glDrawElements(GL_TRIANGLES, (GLsizei)sph.i.size(), GL_UNSIGNED_INT, 0);
                            }
                        }
                    }
                }
                catch (const cv::Exception& e)
                {
                    cerr << "solvePnP: " << e.what() << "\n";
                }
            }
        }
        glfwSwapBuffers(win);
        glfwPollEvents();
        if (glfwGetKey(win, GLFW_KEY_Q) == GLFW_PRESS)
        {
            break;
        }
    }

	// cleanup OpenGL resources and exit
    glDeleteVertexArrays(1, &bgVAO);       
    glDeleteBuffers(1, &bgVBO);
    glDeleteVertexArrays(1, &sphVAO);
    glDeleteBuffers(1, &sphVBO);
    glDeleteBuffers(1, &sphEBO);
    glDeleteVertexArrays(1, &ringVAO);
    glDeleteBuffers(1, &ringVBO);
    glDeleteVertexArrays(1, &satRingVAO);
    glDeleteBuffers(1, &satRingVBO);
    glDeleteBuffers(1, &satRingEBO);
    glDeleteTextures(1, &spaceTex);
    glDeleteProgram(bgProg);
    glDeleteProgram(objProg); glDeleteProgram(ringProg);
    glfwDestroyWindow(win);
    glfwTerminate();
    cap.release();
    return 0;
}