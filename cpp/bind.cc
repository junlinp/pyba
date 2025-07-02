#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <vector>
#include <tuple>
#include "ceres/ceres.h"  // Uncomment when you add Ceres code
#include "ceres/rotation.h"
#include <Eigen/Dense>

namespace py = pybind11;

// Type aliases for clarity
using CameraPoses = std::unordered_map<int64_t, py::array_t<double>>; // 4x4
using Point3Ds = std::unordered_map<int64_t, py::array_t<double>>;    // 3x1
using Observation = std::tuple<int64_t, int64_t, py::array_t<double>>;     // (cam_idx, pt_idx, 2x1)
using Observations = std::vector<Observation>;

Eigen::Matrix3d skew(Eigen::Vector3d v) {
    Eigen::Matrix3d skew_matrix;
    skew_matrix << 0, -v[2], v[1],
                   v[2], 0, -v[0],
                   -v[1], v[0], 0;
    return skew_matrix;
}

Eigen::Matrix3d right_jacobian_rotation(Eigen::Matrix<double, 3, 1> v) {
    Eigen::Matrix3d skew_matrix = skew(v);
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    return I + 0.5 * skew_matrix + (1.0 / 6.0) * skew_matrix * skew_matrix;
}


class ReprojectionError : public ceres::SizedCostFunction<2, 6, 3> {
public:
    ReprojectionError(Eigen::Vector2d observed_keypoint, Eigen::Matrix3d K)
        : observed_keypoint_(observed_keypoint), K_(K) {}

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> camera(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> point_3d(parameters[1]);
        Eigen::Matrix<double, 3, 1> phi = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(camera.data() + 3);

        Eigen::Matrix<double, 3, 1> translation = Eigen::Map<const Eigen::Matrix<double, 3, 1>>(camera.data());
        Eigen::Matrix<double, 3, 3> rotation;
        ceres::AngleAxisToRotationMatrix(camera.data() + 3, rotation.data());

        Eigen::Vector3d point_3d_in_camera = rotation.transpose() * (point_3d - translation);

        Eigen::Vector3d reprojection = K_ * point_3d_in_camera;
        // if (reprojection[2] <= 0) {
        //     std::cout << "camera: " << camera.transpose() << std::endl;
        //     std::cout << "point_3d: " << point_3d.transpose() << std::endl;
        //     std::cout << "reprojection: " << reprojection.transpose() << std::endl;
        //     std::cout << "observed_keypoint: " << observed_keypoint_.transpose() << std::endl;
        //     return false;
        // }

        residuals[0] = reprojection[0] / reprojection[2] - observed_keypoint_[0];
        residuals[1] = reprojection[1] / reprojection[2] - observed_keypoint_[1];

        if (jacobians != nullptr) {
            double x = point_3d_in_camera[0];
            double y = point_3d_in_camera[1];
            double z = point_3d_in_camera[2];
            double z_inv = 1.0 / z;
            double fx = K_(0, 0);
            double fy = K_(1, 1);
            double cx = K_(0, 2);
            double cy = K_(1, 2);
            Eigen::Matrix<double, 2, 3> J_camera_point;
            J_camera_point(0, 0) = fx * z_inv;
            J_camera_point(0, 1) = 0;
            J_camera_point(0, 2) = -fx * x * z_inv * z_inv;
            J_camera_point(1, 0) = 0;
            J_camera_point(1, 1) = fy * z_inv;
            J_camera_point(1, 2) = -fy * y * z_inv * z_inv;
            Eigen::Matrix<double, 3, 3> J_camera_point_to_rotation = rotation.transpose() * skew(point_3d - translation) * right_jacobian_rotation(-phi);

            Eigen::Matrix<double, 3, 3> J_camera_point_to_translation = -rotation.transpose();
            Eigen::Matrix<double, 3, 3> J_camera_point_to_point = rotation.transpose();

            Eigen::Matrix<double, 2, 3> J_rotation = J_camera_point * J_camera_point_to_rotation;
            Eigen::Matrix<double, 2, 3> J_translation = J_camera_point * J_camera_point_to_translation;
            Eigen::Matrix<double, 2, 3> J_point = J_camera_point * J_camera_point_to_point;

            Eigen::Matrix<double, 2, 6> J_camera;
            J_camera.block<2, 3>(0, 3) = J_rotation;
            J_camera.block<2, 3>(0, 0) = J_translation;

            if (jacobians[0] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>(jacobians[0]).noalias() = J_camera;
            }

            if (jacobians[1] != nullptr) {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>(jacobians[1]).noalias() = J_point;
            }
        }
        return true;
    }

    private:
    Eigen::Vector2d observed_keypoint_;
    Eigen::Matrix3d K_;
};

// This is a stub. You will fill in the Ceres logic.
py::tuple ba_solve(
    CameraPoses camera_poses,
    Point3Ds point_3ds,
    Observations observations,
    py::array_t<double> K
) {
    std::cout << "[ba_solve] start" << std::endl;
    std::cout << "[ba_solve] camera_poses: " << camera_poses.size() << std::endl;
    std::cout << "[ba_solve] point_3ds: " << point_3ds.size() << std::endl;
    std::cout << "[ba_solve] observations: " << observations.size() << std::endl;

    ceres::Problem problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

    std::map<int64_t, std::array<double, 6>> camera_parameters;

    int64_t min_cam_indx = std::numeric_limits<int64_t>::max();
    for (const auto& [cam_idx, cam_pose] : camera_poses) {
        std::array<double, 6> camera_parameter;
        auto cam_pose_buf = cam_pose.unchecked<2>();
        Eigen::Matrix4d cam_pose_eigen;
        for (ssize_t i = 0; i < 4; ++i)
            for (ssize_t j = 0; j < 4; ++j)
                cam_pose_eigen(i, j) = cam_pose_buf(i, j);
        
        Eigen::Matrix3d R = cam_pose_eigen.block<3, 3>(0, 0);
        Eigen::Vector3d t = cam_pose_eigen.block<3, 1>(0, 3);
        
        // Debug: Print rotation and translation separately
        std::cout << "Camera " << cam_idx << " rotation matrix:" << std::endl;
        std::cout << R << std::endl;
        std::cout << "Camera " << cam_idx << " translation: " << t.transpose() << std::endl;
        
        camera_parameter[0] = t[0];
        camera_parameter[1] = t[1];
        camera_parameter[2] = t[2];
        ceres::RotationMatrixToAngleAxis(R.data(), camera_parameter.data() + 3);
        camera_parameters[cam_idx] = camera_parameter;
    }

    std::map<int64_t, std::array<double, 3>> point_parameters;
    for (const auto& [pt_idx, pt_3d] : point_3ds) {
        std::cout << "[ba_solve] pt_idx: " << pt_idx << std::endl;
        auto pt_3d_buf = pt_3d.unchecked<1>();
        std::array<double, 3> point_parameter;
        point_parameter[0] = pt_3d_buf(0);
        point_parameter[1] = pt_3d_buf(1);
        point_parameter[2] = pt_3d_buf(2);
        point_parameters[pt_idx] = point_parameter;
        std::cout << "[ba_solve] pt_idx: " << pt_idx << " point_parameter: " << point_parameter[0] << " " << point_parameter[1] << " " << point_parameter[2] << std::endl;
        // std::cout << "point_parameter: " << point_parameter[0] << " " << point_parameter[1] << " " << point_parameter[2] << std::endl;
    }




    ceres::LossFunction* loss_function = new ceres::HuberLoss(2.0);
    auto K_buf = K.unchecked<2>();
    Eigen::Matrix3d K_eigen;
    for (ssize_t i = 0; i < 3; ++i)
        for (ssize_t j = 0; j < 3; ++j)
            K_eigen(i, j) = K_buf(i, j);
    // observations is a list of tuples (cam_idx, pt_idx, 2x1)
    std::cout << "[ba_solve] observations iteration" << std::endl;
    for (const auto& [cam_idx, pt_idx, obs] : observations) {
        std::cout << "[ba_solve] cam_idx: " << cam_idx << std::endl;
        std::cout << "[ba_solve] pt_idx: " << pt_idx << std::endl;
        auto obs_buf = obs.unchecked<1>();
        std::cout << "[ba_solve] unchecked" << std::endl;
        Eigen::Vector2d observed_keypoint(obs_buf(0), obs_buf(1));
        std::cout << "[ba_solve] new ReprojectionError" << std::endl;
        ceres::CostFunction* reprojection_error = new ReprojectionError(observed_keypoint, K_eigen);
        std::cout << "[ba_solve] AddResidualBlock" << std::endl;
        if (camera_parameters.find(cam_idx) == camera_parameters.end()) {
            std::cout << "[ba_solve] camera_parameters.find(cam_idx) == camera_parameters.end()" << std::endl;
            std::cout << "[ba_solve] cam_idx: " << cam_idx << std::endl;
        }
        if (point_parameters.find(pt_idx) == point_parameters.end()) {
            std::cout << "[ba_solve] point_parameters.find(pt_idx) == point_parameters.end()" << std::endl;
            std::cout << "[ba_solve] pt_idx: " << pt_idx << std::endl;
        }
        problem.AddResidualBlock(reprojection_error, loss_function, camera_parameters.at(cam_idx).data(), point_parameters.at(pt_idx).data());
    }

    
    if (min_cam_indx != std::numeric_limits<int64_t>::max()) {
        problem.SetParameterBlockConstant(camera_parameters.at(min_cam_indx).data());
    }

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 1024;
    options.num_threads = 1;


    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    CameraPoses optimized_camera_poses;
    for (const auto& [cam_idx, cam_parameter] : camera_parameters) {
        Eigen::Matrix<double, 4, 4, Eigen::RowMajor> cam_pose_eigen;
        Eigen::Matrix<double, 3, 3> R;
        ceres::AngleAxisToRotationMatrix(cam_parameter.data() + 3, R.data());
        cam_pose_eigen.block<3, 3>(0, 0) = R;
        cam_pose_eigen.block<3, 1>(0, 3) = Eigen::Map<const Eigen::Vector3d>(cam_parameter.data());
        optimized_camera_poses[cam_idx] = py::array_t<double>({4, 4}, cam_pose_eigen.data());
    }

    Point3Ds optimized_point_3ds;
    for (const auto& [pt_idx, pt_parameter] : point_parameters) {
        Eigen::Vector3d pt_3d_eigen = Eigen::Map<const Eigen::Vector3d>(pt_parameter.data());
        optimized_point_3ds[pt_idx] = py::array_t<double>({3}, pt_3d_eigen.data());
    }

    py::dict py_optimized_point_3ds;
    for (const auto& [pt_idx, pt_3d_eigen] : optimized_point_3ds) {
        py_optimized_point_3ds[py::int_(pt_idx)] = py::array_t<double>({3}, pt_3d_eigen.data());
    }
    return py::make_tuple(optimized_camera_poses, py_optimized_point_3ds);
}

PYBIND11_MODULE(pyceres_bind, m) {
    m.doc() = "pybind11 binding for Ceres Solver (template)";

    m.def(
        "ba_solve",
        &ba_solve,
        py::arg("camera_poses"),
        py::arg("point_3ds"),
        py::arg("observations"),
        py::arg("K"),
        R"pbdoc(
            Bundle adjustment solve function.

            Parameters
            ----------
            camera_poses : dict[int, np.ndarray (4x4)]
                Dictionary mapping camera index to 4x4 pose matrix.
            point_3ds : dict[int, np.ndarray (3,)]
                Dictionary mapping point index to 3D point.
            observations : list[tuple[int, int, np.ndarray (2,)]]
                List of (camera_index, point_index, 2D observation).

            Returns
            -------
            tuple
                (optimized_camera_poses, optimized_point_3ds)
        )pbdoc"
    );

    // TODO: Expose your Ceres classes/functions here
    // m.def("solve_bundle_adjustment", &solve_bundle_adjustment, ...);
} 