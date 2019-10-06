"""
deal with multiple people
images with openpose output

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np
import math

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

import pandas as pd 
import os
import glob

import cv2

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')
flags.DEFINE_integer('num_people', None, 
    'Number of people in the image.')

def calibration(proc_params, frameno, i, n):
    import matplotlib.pyplot as plt
    
    img_size = proc_params['img_size']
    undo_scale = 1. / np.array(proc_params['scale'])
    principal_pt = np.array([img_size, img_size]) / 2.
    start_pt = proc_params['start_pt'] - 0.5 * img_size
    final_principal_pt = (principal_pt + start_pt) * undo_scale
    pp = final_principal_pt
    
    p = -1
    if frameno == 1:
        p = i
    else:
        # iterate through the csv files of the previous frame, looking for closest person in that frame.
        dist = float('inf')
        for j in range(n):
            prev_center = np.loadtxt('cali/center/prev_center_%03d_%02d.csv'%(frameno-1, j))
                        
            tmp_dist = distance(prev_center, pp)
            
            if dist > tmp_dist:
                dist = tmp_dist
                p = j
    
    np.savetxt('cali/center/prev_center_%03d_%02d.csv' % (frameno, p), pp, delimiter=',')

    H = np.loadtxt("cali/homography.csv",delimiter=",")
    try : 
        diff = np.loadtxt("cali/diff.csv",delimiter=",")
    except :
        cross_point = np.loadtxt("cali/crosspoint.csv",delimiter=",")
        diff = np.array([pp[0] - cross_point[0],pp[1] - cross_point[1]])
        np.savetxt("cali/diff.csv",diff,delimiter=",")
    
    src = np.array([[pp[0]-diff[0]],[pp[1]-diff[1]],[1]])
    tar = np.matmul(H,src)[:-1].T
        
    return -tar[0][0],-tar[0][1], p


def preprocess_image(img_path, json_path, n):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    
    scales, centers = op_util.get_multiple_bbox(json_path, n)
    
    crops = list()
    proc_params = list()

    for i in range(n):
        crop, proc_param = img_util.scale_and_crop(img, scales[i], centers[i], config.img_size)
        
        # Normalize image to [-1, 1]
        crop = 2 * ((crop / 255.) - 0.5)

        crops.append(crop)
        proc_params.append(proc_param)
    
    return crops, proc_params, img

def main(img_path, json_path, n):
    sess = tf.Session()
    model = RunModel(config, sess=sess)
    
    input_imgs, proc_params, img = preprocess_image(img_path, json_path, n)
    
    R = np.loadtxt("cali/rotate.csv",delimiter=",")
    for i in range(n):
        filename = os.path.splitext(os.path.basename(img_path))[0]
        filename, frame = filename.split('_')[-2:]
        
        input_imgs[i] = np.expand_dims(input_imgs[i], 0)

        joints, verts, cams, joints3d, theta = model.predict(input_imgs[i], get_theta=True)

        x,z,p = calibration(proc_params[i], int(frame), i, n)

        # center = [0.5*img.shape[1],0.5*img.shape[0]]
        # x = (x - center[0])*0.01
        # z = (z - center[1])*0.01
        
        # Rotate
        for k, j in enumerate(joints3d[0]):
            joints3d[0][k] = np.matmul(R, j.T).T
        
        # Translate
        joints3d[0] += np.array([[x,0,z]]*19)

        joints_names = ['Ankle.R_x', 'Ankle.R_y', 'Ankle.R_z',
                   'Knee.R_x', 'Knee.R_y', 'Knee.R_z',
                   'Hip.R_x', 'Hip.R_y', 'Hip.R_z',
                   'Hip.L_x', 'Hip.L_y', 'Hip.L_z',
                   'Knee.L_x', 'Knee.L_y', 'Knee.L_z', 
                   'Ankle.L_x', 'Ankle.L_y', 'Ankle.L_z',
                   'Wrist.R_x', 'Wrist.R_y', 'Wrist.R_z', 
                   'Elbow.R_x', 'Elbow.R_y', 'Elbow.R_z', 
                   'Shoulder.R_x', 'Shoulder.R_y', 'Shoulder.R_z', 
                   'Shoulder.L_x', 'Shoulder.L_y', 'Shoulder.L_z',
                   'Elbow.L_x', 'Elbow.L_y', 'Elbow.L_z',
                   'Wrist.L_x', 'Wrist.L_y', 'Wrist.L_z', 
                   'Neck_x', 'Neck_y', 'Neck_z', 
                   'Head_x', 'Head_y', 'Head_z', 
                   'Nose_x', 'Nose_y', 'Nose_z', 
                   'Eye.L_x', 'Eye.L_y', 'Eye.L_z', 
                   'Eye.R_x', 'Eye.R_y', 'Eye.R_z', 
                   'Ear.L_x', 'Ear.L_y', 'Ear.L_z', 
                   'Ear.R_x', 'Ear.R_y', 'Ear.R_z']
        
        joints_export = pd.DataFrame(joints3d.reshape(1,57), columns=joints_names)
        joints_export.index.name = 'frame'
        
        joints_export.iloc[:, 1::3] = joints_export.iloc[:, 1::3]*-1
        joints_export.iloc[:, 2::3] = joints_export.iloc[:, 2::3]*-1

        hipCenter = joints_export.loc[:][['Hip.R_x', 'Hip.R_y', 'Hip.R_z',
                                      'Hip.L_x', 'Hip.L_y', 'Hip.L_z']]

        joints_export['hip.Center_x'] = hipCenter.iloc[0][::3].sum()/2
        joints_export['hip.Center_y'] = hipCenter.iloc[0][1::3].sum()/2
        joints_export['hip.Center_z'] = hipCenter.iloc[0][2::3].sum()/2
                    
        print("hmr/output/csv/"+os.path.splitext(os.path.basename(img_path))[0]+"_%02d.csv"%p)

        joints_export.to_csv("hmr/output/csv/"+os.path.splitext(os.path.basename(img_path))[0]+"_%02d.csv"%p)

def distance(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

        
def join_csv(n):
    path = 'hmr/output/csv/'                   
    for i in range(n):
        all_files = glob.glob(os.path.join(path, "*_{:02d}.csv".format(i)))

        df_from_each_file = (pd.read_csv(f) for f in sorted(all_files))
        concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)

        concatenated_df['frame'] = concatenated_df.index+1
        concatenated_df.to_csv("hmr/output/csv_joined/csv_joined_{:02d}.csv".format(i), index=False)


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)

    # Using pre-trained model
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    print(config.num_people)
    main(config.img_path, config.json_path, config.num_people)
    

    join_csv(config.num_people)

    print('Result is in hmr/output\n')
