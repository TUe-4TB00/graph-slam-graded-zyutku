import copy
import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)


def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    return result


def minimize_marginals(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.
    best_score = float("inf")
    best_pose = None
    best_landmark = None

    for label, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            trial_graph = copy.deepcopy(graph)
            trial_estimate = copy.deepcopy(initial_estimate)
            trial_graph, trial_estimate = add_pose(trial_graph, trial_estimate, pose_5)
            trial_result = optimize(trial_graph, trial_estimate)
            trial_graph = add_landmark_measurement(trial_graph, trial_result, pose_5, landmark)
            trial_result = optimize(trial_graph, trial_result)

    # TODO: Calculate marginal covariances for the relevant variables and visualize the updated factor graph with covariances
            marginals = gtsam.Marginals(trial_graph, trial_result)
            score = marginals.marginalCovariance(L(landmark)).sum()
            if score < best_score:
                best_score = score
                best_pose = label
                best_landmark = landmark
                best_marginals = marginals
    # The sum of the marginals for each landmark can be computed using marginals.marginalCovariance(L(x)).sum()
    sum_of_marginals = best_marginals.marginalCovariance(L(1)).sum()+ best_marginals.marginalCovariance(L(2)).sum()  
    return best_pose, best_landmark, sum_of_marginals

def minimize_errors(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.
    best_score = float("inf")
    best_pose = None
    best_landmark = None

    for label, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            trial_graph = copy.deepcopy(graph)
            trial_estimate = copy.deepcopy(initial_estimate)
            trial_graph, trial_estimate = add_pose(trial_graph, trial_estimate, pose_5)
            trial_result = optimize(trial_graph, trial_estimate)
            trial_graph = add_landmark_measurement(trial_graph, trial_result, pose_5, landmark)
            trial_result = optimize(trial_graph, trial_result)
# TODO: create a list of errors (each index corresponds to a pose) and add the error of each pose to the list
            marginals = gtsam.Marginals(trial_graph, trial_result)
            list_of_errors = [marginals.marginalCovariance(X(p)).sum() for p in [1, 2, 3]]
            score = sum(list_of_errors)
# TODO: compute the sum of the errors and return it along with the best pose and landmark
            score = sum(list_of_errors)
            if score < best_score:
                best_score = score
                best_pose = label
                best_landmark = landmark
                best_marginals = marginals

    sum_of_errors = best_marginals.marginalCovariance(X(1)).sum() + best_marginals.marginalCovariance(X(2)).sum() + best_marginals.marginalCovariance(X(3)).sum()
    pose_5 = pose_options[best_pose]
    graph, initial_estimate = add_pose(graph, initial_estimate, pose_5)
    return best_pose, best_landmark, sum_of_errors