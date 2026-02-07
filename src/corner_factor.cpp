#include "corner_factor.h"
#include <gtsam/geometry/PinholeCamera.h>

using namespace gtsam;


CornerFactor::CornerFactor(
    Key pose_key,
    const Point3& board_point,
    const Point2& image_point,
    const Cal3_S2& intrinsics,
    const SharedNoiseModel& model)
:
    NoiseModelFactorN(model, pose_key),
    board_point_(board_point),
    image_point_(image_point),
    camera_(Pose3::Identity(), intrinsics)
{}


Vector CornerFactor::evaluateError(
    const Pose3& pose, 
    OptionalMatrixType H1) const 
{
    // Transform point from board space to camera frame
    Matrix36 d_p_d_pose;
    Point3 p = pose.transformFrom(board_point_, d_p_d_pose);

    // Point is behind the camera, return large error and zero Jacobian to avoid local minima
    if (p.z() <= 0) {
        
        if (H1) { *H1 = Matrix26::Zero(); }
        return Vector2::Constant(1000.0);
    }
    
    // Project to pixels
    Matrix23 d_uv_d_p;
    Point2 uv = camera_.project2(p, std::nullopt, d_uv_d_p);

    Vector2 error = uv - image_point_;

    if (H1) {
        *H1 = d_uv_d_p * d_p_d_pose;
    }

    return error;
}




// TipProjectionFactor::TipProjectionFactor(
//     Key tip_pose_key,
//     Vector2 pixels_meas,
//     Cal3_S2 camera_intrinsics,
//     const SharedNoiseModel& model) 
// : 
//     NoiseModelFactorN(model, tip_pose_key),
//     pixels_meas_(pixels_meas),
//     camera_intrinsics_(camera_intrinsics) {}


// Vector TipProjectionFactor::evaluateError(const Pose3& tip_pose, OptionalMatrixType H1) const {
//     Point3 uvz;
//     Matrix36 d_uvz_d_tip_pose;

//     bool behind_camera = project_pose_to_uvz(uvz, tip_pose, camera_intrinsics_, d_uvz_d_tip_pose);
    
//     if (H1) { *H1 = d_uvz_d_tip_pose.block<2,6>(0, 0); }
    
//     if (behind_camera) {
//         return Vector2::Zero();
//     } else {
//         return uvz.head<2>() - pixels_meas_;;
//     }
// }
