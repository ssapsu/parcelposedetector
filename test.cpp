#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <open3d/Open3D.h>
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <map>
#include <string>
#include <iomanip> // For std::setw and std::setprecision for table print
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

void gripperToleranceCalculator(cv::Mat &frame,
                  const Eigen::Vector3f &center,
                  float radius, float height, float cut_height,
                  const Eigen::Matrix3f &rotation_matrix,
                  const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs,
                  const Eigen::Vector3f &t_cg,
                  const Eigen::Vector3f &euler_angles)
{
    // Create 100 points around the circumference of the circles
    const int num_points = 100;
    std::vector<Eigen::Vector3f> top_circle_points(num_points), bottom_circle_points(num_points);

    for (int i = 0; i < num_points; ++i)
    {
        float theta = 2 * M_PI * i / (num_points - 1); // Ensure full circle

        // XZ plane for top and bottom circles
        top_circle_points[i] = Eigen::Vector3f(radius * cos(theta), height / 2, radius * sin(theta));
        bottom_circle_points[i] = Eigen::Vector3f(radius * cos(theta), -height / 2, radius * sin(theta));
    }

    // Combine circle points
    std::vector<Eigen::Vector3f> cylinder_points;
    cylinder_points.insert(cylinder_points.end(), top_circle_points.begin(), top_circle_points.end());
    cylinder_points.insert(cylinder_points.end(), bottom_circle_points.begin(), bottom_circle_points.end());

    // Apply cut height constraint and collect transformed points
    std::vector<cv::Point3f> object_points;
    for (const auto &point : cylinder_points)
    {
        if (point(2) >= cut_height)
        {
            // Rotate and translate points to camera space
            Eigen::Vector3f transformed_point = rotation_matrix * point + center;
            object_points.emplace_back(transformed_point(0), transformed_point(1), transformed_point(2));
        }
    }

    // Project the points onto the image plane
    std::vector<cv::Point2f> projected_points;
    cv::projectPoints(object_points, cv::Vec3d::zeros(), cv::Vec3d::zeros(), camera_matrix, dist_coeffs, projected_points);

    // Compute the gripper's tip in the camera frame
    Eigen::AngleAxisf rollAngle(euler_angles(0), Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle(euler_angles(1), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle(euler_angles(2), Eigen::Vector3f::UnitZ());

    Eigen::Matrix3f R_cg = (rollAngle * pitchAngle * yawAngle).matrix();
    Eigen::Vector3f gripper_tip_camera_frame = R_cg * Eigen::Vector3f::Zero() + t_cg;

    // Extract the gripper's tip coordinates
    float gripper_x = gripper_tip_camera_frame(0);
    float gripper_y = gripper_tip_camera_frame(1);
    float gripper_z = gripper_tip_camera_frame(2);

    // Debugging output for the gripper tip in the camera frame
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Gripper Tip in Camera Frame: X=" << gripper_x
              << ", Y=" << gripper_y
              << ", Z=" << -gripper_z << std::endl;

    // Print cylinder parameters
    std::cout << "Cylinder Center in Camera Frame: X=" << center(0)
              << ", Y=" << center(1)
              << ", Z=" << center(2) << std::endl;
    std::cout << "Cylinder Radius: " << radius << ", Cylinder Height: ±" << (height / 2)
              << ", Cut Height: " << cut_height << std::endl;

    // Check if the gripper tip is inside the cylinder
    bool within_height = (-height / 2 + center(1) + t_cg(1) <= gripper_y && gripper_y <= height / 2 + center(1) + t_cg(1));
    std::cout << "Gripper Tip Height Check: " << (within_height ? "Pass" : "Fail")
              << " (Tip Y=" << gripper_y << " within " << (-height / 2 + center(1) + t_cg(1)) << " to " << (height / 2 + center(1) + t_cg(1)) << ")" << std::endl;

    float distance_from_center = std::sqrt(std::pow(gripper_x - center(0), 2) + std::pow(-gripper_z - center(2), 2));
    bool within_radius = distance_from_center <= radius;
    std::cout << "Gripper Tip Radius Check: " << (within_radius ? "Pass" : "Fail")
              << " (Distance=" << distance_from_center << ", Radius=" << radius << ")" << std::endl;

    bool gripper_inside_cylinder = within_height && within_radius;

    std::cout << "Gripper Inside Cylinder Check: " << (gripper_inside_cylinder ? "Pass" : "Fail") << std::endl;

    // Set color based on whether the gripper is inside the cylinder or not
    cv::Scalar color = gripper_inside_cylinder ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 0, 255); // Blue if inside, Red if outside

    // Draw the projected points of the cylinder
    for (size_t i = 0; i < projected_points.size() - 1; ++i)
    {
        cv::line(frame, projected_points[i], projected_points[i + 1], color, 10);
    }
    // Close the loop for the last point
    if (!projected_points.empty())
    {
        cv::line(frame, projected_points.back(), projected_points[0], color, 2); // Thickness 2 for last line
    }

    // Final result message
    if (gripper_inside_cylinder)
    {
        std::cout << "Gripper tip is inside the cylinder. Ready to grab the parcel!" << std::endl;
    }
    else
    {
        std::cout << "Gripper tip is outside the cylinder. Adjust the position." << std::endl;
    }
}

int main()
{
    // Define MARKER_SIZE and compute translations
    const float MARKER_SIZE = 55.0f;
    const float half_marker_size = MARKER_SIZE / 2.0f;
    const float marker_translation_x = (90.0f - half_marker_size) / 1000.0f;
    const float marker_translation_y = (60.0f - half_marker_size) / 1000.0f;
    const float marker_translation_z = 27.0f / 1000.0f;

    Eigen::Matrix3f R_cam_to_grip;
    R_cam_to_grip << 1, 0, 0,
        0, 0, 1,
        0, 1, 0;

    // Define marker models
    std::map<int, MarkerModel> marker_models = {
        {0, {{marker_translation_x, marker_translation_y, marker_translation_z}, "output.ply"}},
        {1, {{-marker_translation_x, marker_translation_y, marker_translation_z}, "output.ply"}},
        {2, {{-marker_translation_x, -marker_translation_y, marker_translation_z}, "output.ply"}},
        {3, {{marker_translation_x, -marker_translation_y, marker_translation_z}, "output.ply"}}};

    // **Define the drone's pose in the object (now considered world) frame**
    Eigen::Vector3f t_cg(0.0f, -0.02f, -0.115f);
    Eigen::Vector3f euler_angles_deg(-20.0f, 0.0f, 0.0f);
    Eigen::Vector3f euler_angles_rad = euler_angles_deg * static_cast<float>(M_PI) / 180.0f;

    Eigen::AngleAxisf rollAngle(euler_angles_rad(0), Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle(euler_angles_rad(1), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle(euler_angles_rad(2), Eigen::Vector3f::UnitZ());

    Eigen::Matrix3f R_cg = (rollAngle * pitchAngle * yawAngle).matrix();

    // Initialize the Camera object
    Camera camera(0, CameraType::CSI); // You can change to CameraType::WEBCAM for webcam

    cv::Mat camera_matrix = (cv::Mat_<float>(3, 3) << FOCAL_LENGTH_PX, 0.0f, PRINCIPAL_POINT_X,
                             0.0f, FOCAL_LENGTH_PY, PRINCIPAL_POINT_Y,
                             0.0f, 0.0f, 1.0f);

    cv::Mat dist_coeffs = (cv::Mat_<float>(1, 5) << DISTORTION_COEFFS[0], DISTORTION_COEFFS[1], DISTORTION_COEFFS[2], DISTORTION_COEFFS[3], DISTORTION_COEFFS[4]);

    float marker_length = 0.040f; // 40mm converted to meters
    cv::Ptr<cv::aruco::Dictionary> aruco_dict = cv::makePtr<cv::aruco::Dictionary>(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50));
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::makePtr<cv::aruco::DetectorParameters>();
    // set polygonalApproxAccuracy
    parameters->polygonalApproxAccuracyRate = 0.008f;
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
                // Draw detected markers
                cv::aruco::drawDetectedMarkers(frame, corners, ids);

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

                    // **Transform from object (drone) frame to world frame**
                    Eigen::Matrix3f R_co = avg_rotation;
                    Eigen::Vector3f t_co = avg_translation;

                    // Compute object's rotation and translation in gripper frame
                    Eigen::Matrix3f R_go = R_cg * R_co;
                    Eigen::Vector3f t_go = R_cg * t_co + t_cg;

                    // Add the object's position in the world frame to the list
                    object_positions_world.push_back(t_go);

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

                    // Define cylinder parameters
                    float radius = 0.027f;      // Define your cylinder's radius
                    float height = 0.031f;      // Total height of ±0.031 meters (height / 2)
                    float cut_height = -0.016f; // Z-plane cut height for the cylinder

                    // Call the gripperToleranceCalculator function
                    gripperToleranceCalculator(frame, avg_translation, radius, height, cut_height, avg_rotation, camera_matrix, dist_coeffs, t_cg, euler_angles_rad);
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
