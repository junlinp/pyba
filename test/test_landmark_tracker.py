from pyba.landmark_tracker import LandmarkTracker, multiview_triangulation
import numpy as np

if __name__ == "__main__":
    # load data

    landmark_tracker = LandmarkTracker()


    timestamp_1 = 10000000001
    timestamp_2 = 10000000002
    timestamp_3 = 10000000003


    keypoints_1 = np.random.rand(10, 2)
    keypoints_2 = np.random.rand(10, 2)
    keypoints_3 = np.random.rand(10, 2)

    matches_1_2 = np.random.randint(0, 10, (10, 2))
    matches_2_3 = np.random.randint(0, 10, (10, 2))

    landmark_tracker.add_matched_frame(timestamp_1, timestamp_2, keypoints_1, keypoints_2, matches_1_2)
    landmark_tracker.add_matched_frame(timestamp_2, timestamp_3, keypoints_2, keypoints_3, matches_2_3)


    for landmark_id in landmark_tracker.landmarks:
        print(f"landmark_id: {landmark_id}")
        landmark = landmark_tracker.landmarks[landmark_id]
        assert landmark_id in landmark_tracker.landmark_observations
        observations = landmark_tracker.landmark_observations[landmark_id]
        for timestamp, kp_idx in observations.items():
            keypoint2d = landmark_tracker.landmark_keypoints[landmark_id][timestamp][kp_idx]
        
        for timestamp in landmark_tracker.frame_landmarks:
            for kp_idx, landmark_id in landmark_tracker.frame_landmarks[timestamp].items():
                assert landmark_id in landmark_tracker.landmarks, f"{landmark_id} not in {landmark_tracker.landmarks}"
                assert timestamp in landmark_tracker.landmark_keypoints[landmark_id], f"{timestamp} not in {landmark_tracker.landmark_keypoints[landmark_id]}"
                assert kp_idx in landmark_tracker.landmark_keypoints[landmark_id][timestamp], f"{kp_idx} not in {landmark_tracker.landmark_keypoints[landmark_id][timestamp]}"

    
    

    