// src/Camera.h
#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>
#include <string>

enum class CameraType {
    WEBCAM,
    CSI
};

class Camera {
public:
    explicit Camera(int deviceID = 0, CameraType type = CameraType::WEBCAM);
    cv::Mat getFrame();
private:
    cv::VideoCapture cap;
    CameraType cameraType;
    std::string gstreamerPipeline(int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method);
};

#endif // CAMERA_H
