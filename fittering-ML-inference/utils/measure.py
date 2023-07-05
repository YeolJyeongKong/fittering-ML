import sys
import os.path as osp
import json
import sys
import numpy as np
import trimesh
from typing import List, Dict
from scipy.spatial import ConvexHull

import torch

sys.path.append("/home/shin/VScodeProjects/fittering-ML")
import config
from utils.measurement_definitions import *


def load_face_segmentation(segmentation_path):
    try:
        with open(segmentation_path, 'r') as f:
            face_segmentation = json.load(f)
    except FileNotFoundError:
        sys.exit(f"No such file - {segmentation_path}")

    return face_segmentation




class MeasureVerts:
    face_segmentation = load_face_segmentation(config.SEGMENTATION_PATH)
    faces = np.load(config.SMPL_FACES_PATH)

    measurement_names = MeasurementDefinitions.possible_measurements
    length_definitions = MeasurementDefinitions.LENGTHS
    circumf_definitions = MeasurementDefinitions.CIRCUMFERENCES
    measurement_types = MeasurementDefinitions.measurement_types
    circumf_2_bodypart = MeasurementDefinitions.CIRCUMFERENCE_TO_BODYPARTS
    
    circumference_type = MeasurementType.CIRCUMFERENCE
    length_type = MeasurementType.LENGTH

    cached_visualizations = {"LENGTHS":{}, "CIRCUMFERENCES":{}}

    landmarks = LANDMARK_INDICES

    @staticmethod
    def _get_dist(verts: np.ndarray) -> float:
        verts_distances = np.linalg.norm(verts[:, 1] - verts[:, 0],axis=1)
        distance = np.sum(verts_distances)
        distance_cm = distance * 100 # convert to cm
        return distance_cm

    @classmethod
    def measure_length(cls, measurement_name: str, verts):
        measurement_landmarks_inds = cls.length_definitions[measurement_name]

        landmark_points = []
        for i in range(2):
            if isinstance(measurement_landmarks_inds[i],tuple):
                # if touple of indices for landmark, take their average
                lm = (verts[measurement_landmarks_inds[i][0]] + 
                            verts[measurement_landmarks_inds[i][1]]) / 2
            else:
                lm = verts[measurement_landmarks_inds[i]]
            
            landmark_points.append(lm)

        landmark_points = np.vstack(landmark_points)[None,...]

        return cls._get_dist(landmark_points)

    @classmethod
    def measure_circumference(cls, measurement_name: str, verts, joints):
        measurement_definition = cls.circumf_definitions[measurement_name]
        circumf_landmarks = measurement_definition["LANDMARKS"]
        circumf_landmark_indices = [cls.landmarks[l_name] for l_name in circumf_landmarks]
        circumf_n1, circumf_n2 = cls.circumf_definitions[measurement_name]["JOINTS"]
        circumf_n1, circumf_n2 = JOINT2IND[circumf_n1], JOINT2IND[circumf_n2]
        
        plane_origin = np.mean(verts[circumf_landmark_indices,:],axis=0)
        plane_normal = joints[circumf_n1,:] - joints[circumf_n2,:]

        mesh = trimesh.Trimesh(vertices=verts, faces=cls.faces)

        # new version            
        slice_segments, sliced_faces = trimesh.intersections.mesh_plane(mesh, 
                                plane_normal=plane_normal, 
                                plane_origin=plane_origin, 
                                return_faces=True) # (N, 2, 3), (N,)
        
        slice_segments = cls._filter_body_part_slices(slice_segments,
                                                    sliced_faces,
                                                    measurement_name)
        
        slice_segments_hull = cls._circumf_convex_hull(slice_segments)

        return cls._get_dist(slice_segments_hull)

    @classmethod
    def _filter_body_part_slices(cls, slice_segments:np.ndarray, sliced_faces:np.ndarray,
                                measurement_name: str):
        if measurement_name in cls.circumf_2_bodypart.keys():

            body_parts = cls.circumf_2_bodypart[measurement_name]

            if isinstance(body_parts,list):
                body_part_faces = [face_index for body_part in body_parts 
                                    for face_index in cls.face_segmentation[body_part]]
            else:
                body_part_faces = cls.face_segmentation[body_parts]

            N_sliced_faces = sliced_faces.shape[0]

            keep_segments = []
            for i in range(N_sliced_faces):
                if sliced_faces[i] in body_part_faces:
                    keep_segments.append(i)

            return slice_segments[keep_segments]

        else:
            return slice_segments
    
    @staticmethod
    def _circumf_convex_hull(slice_segments: np.ndarray):
        '''
        Cretes convex hull from 3D points
        :param slice_segments: np.ndarray, dim N x 2 x 3 representing N 3D segments

        Returns:
        :param slice_segments_hull: np.ndarray, dim N x 2 x 3 representing N 3D segments
                                    that form the convex hull
        '''

        # stack all points in N x 3 array
        merged_segment_points = np.concatenate(slice_segments)
        unique_segment_points = np.unique(merged_segment_points,
                                            axis=0)

        # points lie in plane -- find which ax of x,y,z is redundant
        redundant_plane_coord = np.argmin(np.max(unique_segment_points,axis=0) - 
                                            np.min(unique_segment_points,axis=0) )
        non_redundant_coords = [x for x in range(3) if x!=redundant_plane_coord]

        # create convex hull
        hull = ConvexHull(unique_segment_points[:,non_redundant_coords])
        segment_point_hull_inds = hull.simplices.reshape(-1)

        slice_segments_hull = unique_segment_points[segment_point_hull_inds]
        slice_segments_hull = slice_segments_hull.reshape(-1,2,3)

        return slice_segments_hull

    @classmethod
    def measure(cls, verts, joints):
        measurements = {}
        for m_name in cls.measurement_names:
            if m_name in measurements:
                pass

            if cls.measurement_types[m_name] == cls.length_type:
                value = cls.measure_length(m_name, verts)
                measurements[m_name] = value

            elif cls.measurement_types[m_name] == cls.circumference_type:
                value = cls.measure_circumference(m_name, verts, joints)
                measurements[m_name] = value

            else:
                raise Exception("none possible measurements")
        return measurements

    @classmethod
    def verts2meas(cls, verts, joints):
        joints = joints.detach().cpu().numpy()
        verts = verts.detach().cpu().numpy()

        return cls.measure(verts, joints)
 