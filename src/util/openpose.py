"""
Script to convert openpose output into bbox
"""
import json
import numpy as np


def read_json(json_path):
    with open(json_path) as f:
        data = json.load(f)
    kps = []
    for people in data['people']:
        kp = np.array(people['pose_keypoints_2d']).reshape(-1, 3)
        kps.append(kp)
    return kps


def get_bbox(json_path, vis_thr=0.2):
    kps = read_json(json_path)
    # Pick the most confident detection.
    scores = [np.mean(kp[kp[:, 2] > vis_thr, 2]) for kp in kps]
    kp = kps[np.argmax(scores)]
    vis = kp[:, 2] > vis_thr
    vis_kp = kp[vis, :2]
    min_pt = np.min(vis_kp, axis=0)
    max_pt = np.max(vis_kp, axis=0)
    person_height = np.linalg.norm(max_pt - min_pt)
    if person_height == 0:
        print('bad!')
        import ipdb
        ipdb.set_trace()
    center = (min_pt + max_pt) / 2.
    scale = 150. / person_height

    return scale, center

# return multiple scales, centers
def get_multiple_bbox(json_path, n = 1, vis_thr=0.2):    
    kps = read_json(json_path)
    # Pick top n detections.
    scores = np.array([np.mean(kp[kp[:, 2] > vis_thr, 2]) for kp in kps])
    scores = np.array(map(lambda x: 0 if np.isnan(x) else x,scores))
    kp = [kps[s] for s in scores.argsort()[-n:][::-1]]
    
    scales = list()
    centers = list()
    for i in range(n):
        try:
            vis = kp[i][:,2] > vis_thr
            vis_kp = kp[i][vis, :2]
            min_pt = np.min(vis_kp, axis=0)
            max_pt = np.max(vis_kp, axis=0)
        except:
            print("Too Low confidence!",i)
            
        person_height = np.linalg.norm(max_pt - min_pt)
        if person_height == 0:
            print('bad!')
            import ipdb; ipdb.set_trace()
            ipdb.set_trace()
        center = (min_pt + max_pt) / 2.
        scale = 150. / person_height
        
        centers.append(center)
        scales.append(scale)

    return scales, centers
