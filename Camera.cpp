// src/Camera.cpp
#include "Camera.h"
#include "CameraConstants.h"

Camera::Camera(int deviceID, CameraType type)
    : cameraType(type) {
    if (cameraType == CameraType::CSI) {
        // CSI 카메라를 위한 GStreamer 파이프라인 생성
        int capture_width = SENSOR_RESOLUTION_X;
        int capture_height = SENSOR_RESOLUTION_Y;
        int display_width = SENSOR_RESOLUTION_X;
        int display_height = SENSOR_RESOLUTION_Y;
        int framerate = SENSOR_FPS;
        int flip_method = 0;

        std::string pipeline = gstreamerPipeline(
            capture_width,
            capture_height,
            display_width,
            display_height,
            framerate,
            flip_method
        );
        cap = cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);
    } else {
        // 일반 웹캠 초기화
        cap.open(deviceID);
    }

    if (!cap.isOpened()) {
        throw std::runtime_error("카메라를 열 수 없습니다.");
    }
}

cv::Mat Camera::getFrame() {
    cv::Mat frame;
    cap.read(frame);
    // cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
    return frame;
}

std::string Camera::gstreamerPipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) +
           ", height=(int)" + std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" +
           std::to_string(display_width) + ", height=(int)" + std::to_string(display_height) +
           ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}
