#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <open3d/Open3D.h>
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <map>
#include <string>
#include <iomanip> // For std::setw for table print
#include "Camera.h"
#include "CameraConstants.h"

// Function to compute average translation
Eigen::Vector3f computeAverageTranslation(const std::vector<Eigen::Vector3f> &translations)
{
    Eigen::Vector3f avg_translation = Eigen::Vector3f::Zero();
    for (const auto &trans : translations)
    {
        avg_translation += trans;
    }
    avg_translation /= translations.size();
    return avg_translation;
}

struct MarkerModel
{
    Eigen::Vector3f translation;
    std::string model_path;
};

std::vector<Eigen::Vector3f> loadModel(const std::string &model_path, float scale_factor = 1.0f / 1000.0f)
{
    auto mesh = open3d::io::CreateMeshFromFile(model_path);
    if (!mesh || mesh->vertices_.empty())
    {
        std::cerr << "Error: Unable to load the model from " << model_path << ". Please check the file path." << std::endl;
        return std::vector<Eigen::Vector3f>();
    }

    auto vertices = mesh->vertices_;
    std::vector<Eigen::Vector3f> transformed_vertices;

    for (const auto &vertex : vertices)
    {
        Eigen::Vector3f v(vertex(0), vertex(2), vertex(1)); // Swap Y and Z
        v *= scale_factor;
        transformed_vertices.push_back(v);
    }

    std::cout << "Loaded model with " << transformed_vertices.size() << " vertices." << std::endl; // Debugging line
    return transformed_vertices;
}

void drawAxis(cv::Mat &frame, const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
              const cv::Vec3d &rvec, const cv::Vec3d &tvec, float axis_length = 0.05f)
{
    cv::drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_length);
}

Eigen::Matrix3f averageRotationMatrices(const std::vector<Eigen::Matrix3f> &rot_matrices)
{
    std::vector<Eigen::Quaternionf> quaternions;
    for (const auto &rot : rot_matrices)
    {
        Eigen::Quaternionf q(rot);
        quaternions.push_back(q);
    }

    Eigen::Quaternionf avg_q(0, 0, 0, 0);
    for (const auto &q : quaternions)
    {
        avg_q.coeffs() += q.coeffs();
    }
    avg_q.coeffs() /= quaternions.size();
    avg_q.normalize();

    return avg_q.toRotationMatrix();
}

int main()
{
    // Define MARKER_SIZE and compute translations
    const float MARKER_SIZE = 55.0f;
    const float half_marker_size = MARKER_SIZE / 2.0f;
    const float marker_translation_x = (90.0f - half_marker_size) / 1000.0f;
    const float marker_translation_y = (60.0f - half_marker_size) / 1000.0f;
    const float marker_translation_z = 27.0f / 1000.0f;

    // Define marker models
    std::map<int, MarkerModel> marker_models = {
        {0, {{marker_translation_x, marker_translation_y, marker_translation_z}, "output.ply"}},
        {1, {{-marker_translation_x, marker_translation_y, marker_translation_z}, "output.ply"}},
        {2, {{-marker_translation_x, -marker_translation_y, marker_translation_z}, "output.ply"}},
        {3, {{marker_translation_x, -marker_translation_y, marker_translation_z}, "output.ply"}}};

    // **Define the camera's pose in the world frame**
    Eigen::Vector3f t_wc(0.0f, -0.014f, -0.10f);
    Eigen::Vector3f euler_angles_deg(-20.0f, 0.0f, 0.0f);
    Eigen::Vector3f euler_angles_rad = euler_angles_deg * static_cast<float>(M_PI) / 180.0f;

    Eigen::AngleAxisf rollAngle(euler_angles_rad(0), Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle(euler_angles_rad(1), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle(euler_angles_rad(2), Eigen::Vector3f::UnitZ());

    Eigen::Matrix3f R_wc = (rollAngle * pitchAngle * yawAngle).matrix();

    // Initialize the Camera object
    Camera camera(0, CameraType::CSI); // You can change to CameraType::WEBCAM for webcam

    cv::Mat camera_matrix = (cv::Mat_<float>(3, 3) << FOCAL_LENGTH_PX, 0.0f, PRINCIPAL_POINT_X,
                             0.0f, FOCAL_LENGTH_PY, PRINCIPAL_POINT_Y,
                             0.0f, 0.0f, 1.0f);

    cv::Mat dist_coeffs = (cv::Mat_<float>(1, 5) << DISTORTION_COEFFS[0], DISTORTION_COEFFS[1], DISTORTION_COEFFS[2], DISTORTION_COEFFS[3], DISTORTION_COEFFS[4]);

    float marker_length = 0.040f; // 40mm converted to meters
    cv::Ptr<cv::aruco::Dictionary> aruco_dict = cv::makePtr<cv::aruco::Dictionary>(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50));
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::makePtr<cv::aruco::DetectorParameters>();

    // Load the model once
    std::vector<Eigen::Vector3f> model_vertices = loadModel(marker_models[0].model_path);

    // Define the target point in the model's local coordinate system
    Eigen::Vector3f target_point(0.0f, 0.0f, 0.0f);

    // List to store the last 20 object positions in the world frame
    std::vector<Eigen::Vector3f> object_positions_world;
    const size_t max_positions = 20;

    // Initialize FPS calculation
    double tick_frequency = cv::getTickFrequency();
    double prev_tick = cv::getTickCount();
    double fps = 0;

    try
    {
        cv::namedWindow("ObjectPoseEstimation", cv::WINDOW_AUTOSIZE);
        while (true)
        {
            // Calculate FPS
            double current_tick = cv::getTickCount();
            double time_per_frame = (current_tick - prev_tick) / tick_frequency;
            prev_tick = current_tick;
            fps = 1.0 / time_per_frame;

            // Use the Camera class to get the frame
            cv::Mat frame = camera.getFrame();
            if (frame.empty())
            {
                std::cerr << "Cannot read frame from camera." << std::endl;
                break;
            }

            // Display FPS in the top-right corner of the frame
            std::string fps_text = "FPS: " + std::to_string(int(fps));
            int baseline;
            cv::Size text_size = cv::getTextSize(fps_text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
            cv::Point fps_position(frame.cols - text_size.width - 10, 30);
            cv::putText(frame, fps_text, fps_position, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

            // ArUco marker detection
            std::vector<int> ids;
            std::vector<std::vector<cv::Point2f>> corners;
            cv::aruco::detectMarkers(frame, aruco_dict, corners, ids, parameters);

            if (!ids.empty())
            {
                // Lists to store transformations
                std::vector<Eigen::Vector3f> translations;
                std::vector<Eigen::Matrix3f> rotations;

                // Estimate pose for each marker
                std::vector<cv::Vec3d> rvecs, tvecs;
                cv::aruco::estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs, rvecs, tvecs);

                for (size_t i = 0; i < ids.size(); ++i)
                {
                    int marker_id = ids[i];

                    if (marker_models.find(marker_id) != marker_models.end())
                    {
                        // Draw axes
                        drawAxis(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.05f);

                        // Calculate rotation matrix
                        cv::Mat rot_mat;
                        cv::Rodrigues(rvecs[i], rot_mat);

                        Eigen::Matrix3f rotation_matrix;
                        for (int row = 0; row < 3; ++row)
                            for (int col = 0; col < 3; ++col)
                                rotation_matrix(row, col) = static_cast<float>(rot_mat.at<double>(row, col));

                        // Add bias
                        Eigen::Vector3f bias = marker_models[marker_id].translation;

                        // Transform bias to camera coordinate system
                        Eigen::Vector3f rotated_bias = rotation_matrix * bias;

                        // Get the marker translation
                        Eigen::Vector3f marker_translation = Eigen::Vector3f(static_cast<float>(tvecs[i][0]), static_cast<float>(tvecs[i][1]), static_cast<float>(tvecs[i][2])) + rotated_bias;

                        // Store transformations
                        translations.push_back(marker_translation);
                        rotations.push_back(rotation_matrix);
                    }
                }

                // Process transformations if available
                if (!translations.empty() && !rotations.empty())
                {
                    Eigen::Vector3f avg_translation = computeAverageTranslation(translations);
                    Eigen::Matrix3f avg_rotation = averageRotationMatrices(rotations);

                    // Transform object pose from camera frame to world frame
                    Eigen::Matrix3f R_co = avg_rotation;
                    Eigen::Vector3f t_co = avg_translation;

                    Eigen::Matrix3f R_wo = R_wc * R_co;
                    Eigen::Vector3f t_wo = R_wc * t_co + t_wc;

                    // Add the object's position in the world frame to the list
                    object_positions_world.push_back(t_wo);

                    // Keep only the last 20 positions
                    if (object_positions_world.size() > max_positions)
                    {
                        object_positions_world.erase(object_positions_world.begin());
                    }

                    // Calculate the mean of the last 20 positions
                    Eigen::Vector3f mean_position = computeAverageTranslation(object_positions_world);

                    // Print results in a table-like format
                    std::cout << std::setw(15) << "Object ID"
                              << std::setw(15) << "Position X"
                              << std::setw(15) << "Position Y"
                              << std::setw(15) << "Position Z" << std::endl;
                    std::cout << std::string(60, '-') << std::endl;
                    std::cout << std::setw(15) << "Mean Position"
                              << std::setw(15) << mean_position(0)
                              << std::setw(15) << mean_position(1)
                              << std::setw(15) << mean_position(2) << std::endl;

                    // Display the mean position on the frame
                    std::string mean_pos_text = "Mean Position: X=" + std::to_string(mean_position(0)) +
                                                ", Y=" + std::to_string(mean_position(1)) + ", Z=" + std::to_string(mean_position(2));
                    cv::putText(frame, mean_pos_text, cv::Point(10, frame.rows - 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

                    // Transform the target point using the average rotation and translation in camera frame
                    Eigen::Vector3f transformed_target_point = avg_rotation * target_point + avg_translation;

                    // Project the transformed target point onto the image plane
                    std::vector<cv::Point3f> object_points = {cv::Point3f(transformed_target_point(0), transformed_target_point(1), transformed_target_point(2))};
                    std::vector<cv::Point2f> projected_points;
                    cv::projectPoints(object_points, cv::Vec3d::zeros(), cv::Vec3d::zeros(), camera_matrix, dist_coeffs, projected_points);

                    // Get the 2D coordinates of the projected point
                    cv::Point2f highlight_point = projected_points[0];

                    // Draw the point on the image
                    cv::circle(frame, highlight_point, 10, cv::Scalar(0, 0, 255), -1); // Red filled circle

                    // **Project and Draw the Model Vertices**
                    if (!model_vertices.empty())
                    {
                        std::vector<cv::Point3f> transformed_vertices;
                        for (const auto &vertex : model_vertices)
                        {
                            Eigen::Vector3f transformed_vertex = avg_rotation * vertex + avg_translation;
                            transformed_vertices.emplace_back(transformed_vertex(0), transformed_vertex(1), transformed_vertex(2));
                        }

                        // Project model vertices to image coordinates
                        std::vector<cv::Point2f> image_points;
                        cv::projectPoints(transformed_vertices, cv::Vec3d::zeros(), cv::Vec3d::zeros(), camera_matrix, dist_coeffs, image_points);

                        // Draw model vertices on the frame
                        for (const auto &pt : image_points)
                        {
                            cv::circle(frame, pt, 3, cv::Scalar(0, 255, 0), -1); // Green dots for vertices
                        }
                    }
                }
            }

            cv::imshow("ObjectPoseEstimation", frame);

            char key = (char)cv::waitKey(10);
            if (key == 27 || key == 'q')
            {
                break;
            }
        }
        cv::destroyAllWindows();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Camera initialization error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
