//clang++ -o build/test_device cpp/test_device.cpp -Iegl_include -lpthread -ldl -lGL -lEGL -std=c++11
// https://github.com/fxia22/egl_example/blob/master/test_device.cpp
#include <stdio.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>

struct EGLInternalData2 {
    bool m_isInitialized;

    int m_windowWidth;
    int m_windowHeight;
    int m_renderDevice;

    EGLBoolean success;
    EGLint num_configs;
    EGLConfig egl_config;
    EGLSurface egl_surface;
    EGLContext egl_context;
    EGLDisplay egl_display;

    EGLInternalData2()
    : m_isInitialized(false),
    m_windowWidth(0),
    m_windowHeight(0) {}
};


int main(int argc, char ** argv) {
    // Load extensions
    // https://github.com/pmh47/dirt/blob/master/csrc/gl_common.h#L42
    auto const eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC) eglGetProcAddress("eglQueryDevicesEXT");
    auto const eglQueryDeviceAttribEXT = (PFNEGLQUERYDEVICEATTRIBEXTPROC) eglGetProcAddress("eglQueryDeviceAttribEXT");
    auto const eglGetPlatformDisplayEXT = (PFNEGLGETPLATFORMDISPLAYEXTPROC) eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!eglQueryDevicesEXT || !eglQueryDeviceAttribEXT || !eglGetPlatformDisplayEXT) {
        fprintf(stderr, "extensions: eglQueryDevicesEXT, eglQueryDeviceAttribEXT and eglGetPlatformDisplayEXT not available.\n");
        exit(EXIT_FAILURE);
    }

    EGLInternalData2* m_data = new EGLInternalData2();

    m_data->m_renderDevice = atoi(argv[1]);

    printf("test device: %d\n", m_data->m_renderDevice);


    int m_windowWidth;
    int m_windowHeight;
    int m_renderDevice;

    EGLBoolean success;
    EGLint num_configs;
    EGLConfig egl_config;
    EGLSurface egl_surface;
    EGLContext egl_context;
    EGLDisplay egl_display;

    m_windowWidth = 256;
    m_windowHeight = 256;
    m_renderDevice = -1;

    EGLint egl_config_attribs[] = {
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_DEPTH_SIZE, 8,
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE};

    EGLint egl_pbuffer_attribs[] = {
        EGL_WIDTH, m_windowWidth,
        EGL_HEIGHT, m_windowHeight,
        EGL_NONE,
    };

    // Query EGL Devices
    const int max_devices = 32;
    EGLDeviceEXT egl_devices[max_devices];
    EGLint num_devices = 0;
    EGLint egl_error = eglGetError();
    if (!eglQueryDevicesEXT(max_devices, egl_devices, &num_devices) ||
        egl_error != EGL_SUCCESS) {
        printf("eglQueryDevicesEXT Failed.\n");
        m_data->egl_display = EGL_NO_DISPLAY;
    }

    printf("number of devices found %d\n", num_devices);

    // Set display
    int major, minor;
    EGLDisplay display = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, egl_devices[m_data->m_renderDevice], NULL);
    if (eglGetError() == EGL_SUCCESS && display != EGL_NO_DISPLAY) {

        EGLBoolean initialized = eglInitialize(display, &major, &minor);
        if (eglGetError() == EGL_SUCCESS && initialized == EGL_TRUE) {
            m_data->egl_display = display;
        }
    }

    if (!eglInitialize(m_data->egl_display, NULL, NULL)) {
        fprintf(stderr, "Unable to initialize EGL\n");
        exit(EXIT_FAILURE);
    }

    printf("(egl_renderer) Loaded EGL version: %d.%d.\n", major, minor);

    m_data->success = eglBindAPI(EGL_OPENGL_API);
    if (!m_data->success) {
        // TODO: Properly handle this error (requires change to default window
        // API to change return on all window types to bool).
        fprintf(stderr, "Failed to bind OpenGL API.\n");
        exit(EXIT_FAILURE);
    }

    m_data->success = eglChooseConfig(m_data->egl_display, egl_config_attribs, &m_data->egl_config, 1, &m_data->num_configs);
    if (!m_data->success) {
        // TODO: Properly handle this error (requires change to default window
        // API to change return on all window types to bool).
        fprintf(stderr, "Failed to choose config (eglError: %d)\n", eglGetError());
        exit(EXIT_FAILURE);
    }
    if (m_data->num_configs != 1) {
        fprintf(stderr, "Didn't get exactly one config, but %d\n", m_data->num_configs);
        exit(EXIT_FAILURE);
    }

    m_data->egl_surface = eglCreatePbufferSurface(m_data->egl_display, m_data->egl_config, egl_pbuffer_attribs);
    if (m_data->egl_surface == EGL_NO_SURFACE) {
        fprintf(stderr, "Unable to create EGL surface (eglError: %d)\n", eglGetError());
        exit(EXIT_FAILURE);
    }


    m_data->egl_context = eglCreateContext(m_data->egl_display, m_data->egl_config, EGL_NO_CONTEXT, NULL);
    if (!m_data->egl_context) {
        fprintf(stderr, "Unable to create EGL context (eglError: %d)\n",eglGetError());
        exit(EXIT_FAILURE);
    }

    m_data->success = eglMakeCurrent(m_data->egl_display, m_data->egl_surface, m_data->egl_surface, m_data->egl_context);
    if (!m_data->success) {
        fprintf(stderr, "Failed to make context current (eglError: %d)\n", eglGetError());
        exit(EXIT_FAILURE);
    }

    const GLubyte* ven = glGetString(GL_VENDOR);
    printf("GL_VENDOR=%s\n", ven);

    const GLubyte* ren = glGetString(GL_RENDERER);
    printf("GL_RENDERER=%s\n", ren);
    const GLubyte* ver = glGetString(GL_VERSION);
    printf("GL_VERSION=%s\n", ver);
    const GLubyte* sl = glGetString(GL_SHADING_LANGUAGE_VERSION);
    printf("GL_SHADING_LANGUAGE_VERSION=%s\n", sl);

    return 0;
}
