#pragma once

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholeCamera.h>


class CornerFactor: public gtsam::NoiseModelFactorN<gtsam::Pose3> {
    using gtsam::NoiseModelFactorN<gtsam::Pose3>::evaluateError;

public:
    CornerFactor(
        gtsam::Key pose_key,
        const gtsam::Point3& board_point,
        const gtsam::Point2& image_point,
        const gtsam::Cal3_S2& intrinsics,
        const gtsam::SharedNoiseModel& model);
    
    gtsam::Vector evaluateError(
        const gtsam::Pose3& pose, 
        gtsam::OptionalMatrixType H1) const override;

private:
    gtsam::Point3 board_point_;
    gtsam::Point2 image_point_;
    gtsam::PinholeCamera<gtsam::Cal3_S2> camera_;
};
