import sys
import os.path as osp
import json
import sys
import numpy as np
import trimesh
from typing import List, Dict
from scipy.spatial import ConvexHull
import plotly
import plotly.graph_objects as go
import plotly.express as px

import torch

sys.path.append("/home/shin/VScodeProjects/fittering-ML")
from extras import paths
from src.utils.measurement_definitions import *


def load_face_segmentation(segmentation_path):
    try:
        with open(segmentation_path, "r") as f:
            face_segmentation = json.load(f)
    except FileNotFoundError:
        sys.exit(f"No such file - {segmentation_path}")

    return face_segmentation


class MeasureVerts:
    face_segmentation = load_face_segmentation(paths.SEGMENTATION_PATH)
    faces = np.load(paths.SMPL_FACES_PATH)

    measurement_names = MeasurementDefinitions.possible_measurements
    length_definitions = MeasurementDefinitions.LENGTHS
    circumf_definitions = MeasurementDefinitions.CIRCUMFERENCES
    measurement_types = MeasurementDefinitions.measurement_types
    circumf_2_bodypart = MeasurementDefinitions.CIRCUMFERENCE_TO_BODYPARTS

    circumference_type = MeasurementType.CIRCUMFERENCE
    length_type = MeasurementType.LENGTH

    cached_visualizations = {"LENGTHS": {}, "CIRCUMFERENCES": {}}

    landmarks = LANDMARK_INDICES

    @staticmethod
    def _get_dist(verts: np.ndarray) -> float:
        verts_distances = np.linalg.norm(verts[:, 1] - verts[:, 0], axis=1)
        distance = np.sum(verts_distances)
        distance_cm = distance * 100  # convert to cm
        return distance_cm

    @classmethod
    def measure_length(cls, measurement_name: str, verts):
        measurement_landmarks_inds = cls.length_definitions[measurement_name]

        landmark_points = []
        for i in range(2):
            if isinstance(measurement_landmarks_inds[i], tuple):
                # if touple of indices for landmark, take their average
                lm = (
                    verts[measurement_landmarks_inds[i][0]]
                    + verts[measurement_landmarks_inds[i][1]]
                ) / 2
            else:
                lm = verts[measurement_landmarks_inds[i]]

            landmark_points.append(lm)

        landmark_points = np.vstack(landmark_points)[None, ...]

        return cls._get_dist(landmark_points)

    @classmethod
    def measure_circumference(cls, measurement_name: str, verts, joints):
        measurement_definition = cls.circumf_definitions[measurement_name]
        circumf_landmarks = measurement_definition["LANDMARKS"]
        circumf_landmark_indices = [
            cls.landmarks[l_name] for l_name in circumf_landmarks
        ]
        circumf_n1, circumf_n2 = cls.circumf_definitions[measurement_name]["JOINTS"]
        circumf_n1, circumf_n2 = JOINT2IND[circumf_n1], JOINT2IND[circumf_n2]

        plane_origin = np.mean(verts[circumf_landmark_indices, :], axis=0)
        plane_normal = joints[circumf_n1, :] - joints[circumf_n2, :]

        mesh = trimesh.Trimesh(vertices=verts, faces=cls.faces)

        # new version
        slice_segments, sliced_faces = trimesh.intersections.mesh_plane(
            mesh,
            plane_normal=plane_normal,
            plane_origin=plane_origin,
            return_faces=True,
        )  # (N, 2, 3), (N,)

        slice_segments = cls._filter_body_part_slices(
            slice_segments, sliced_faces, measurement_name
        )

        slice_segments_hull = cls._circumf_convex_hull(slice_segments)

        return cls._get_dist(slice_segments_hull)

    @classmethod
    def _filter_body_part_slices(
        cls, slice_segments: np.ndarray, sliced_faces: np.ndarray, measurement_name: str
    ):
        if measurement_name in cls.circumf_2_bodypart.keys():
            body_parts = cls.circumf_2_bodypart[measurement_name]

            if isinstance(body_parts, list):
                body_part_faces = [
                    face_index
                    for body_part in body_parts
                    for face_index in cls.face_segmentation[body_part]
                ]
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
        """
        Cretes convex hull from 3D points
        :param slice_segments: np.ndarray, dim N x 2 x 3 representing N 3D segments

        Returns:
        :param slice_segments_hull: np.ndarray, dim N x 2 x 3 representing N 3D segments
                                    that form the convex hull
        """

        # stack all points in N x 3 array
        merged_segment_points = np.concatenate(slice_segments)
        unique_segment_points = np.unique(merged_segment_points, axis=0)

        # points lie in plane -- find which ax of x,y,z is redundant
        redundant_plane_coord = np.argmin(
            np.max(unique_segment_points, axis=0)
            - np.min(unique_segment_points, axis=0)
        )
        non_redundant_coords = [x for x in range(3) if x != redundant_plane_coord]

        # create convex hull
        hull = ConvexHull(unique_segment_points[:, non_redundant_coords])
        segment_point_hull_inds = hull.simplices.reshape(-1)

        slice_segments_hull = unique_segment_points[segment_point_hull_inds]
        slice_segments_hull = slice_segments_hull.reshape(-1, 2, 3)

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

    @classmethod
    def create_mesh_plot(cls, verts: np.ndarray):
        """
        Visualize smpl body mesh.
        :param verts: np.array (N,3) of vertices
        :param faces: np.array (F,3) of faces connecting the vertices

        Return:
        plotly Mesh3d object for plotting
        """
        mesh_plot = go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            color="gray",
            hovertemplate="<i>Index</i>: %{text}",
            text=[i for i in range(verts.shape[0])],
            # i, j and k give the vertices of triangles
            i=cls.faces[:, 0],
            j=cls.faces[:, 1],
            k=cls.faces[:, 2],
            opacity=0.6,
            name="body",
        )
        return mesh_plot

    @staticmethod
    def create_joint_plot(joints: np.ndarray):
        return go.Scatter3d(
            x=joints[:, 0],
            y=joints[:, 1],
            z=joints[:, 2],
            mode="markers",
            marker=dict(size=8, color="black", opacity=1, symbol="cross"),
            name="joints",
        )

    @classmethod
    def create_wireframe_plot(cls, verts: np.ndarray):
        """
        Given vertices and faces, creates a wireframe of plotly segments.
        Used for visualizing the wireframe.

        :param verts: np.array (N,3) of vertices
        :param faces: np.array (F,3) of faces connecting the verts
        """
        i = cls.faces[:, 0]
        j = cls.faces[:, 1]
        k = cls.faces[:, 2]

        triangles = np.vstack((i, j, k)).T

        x = verts[:, 0]
        y = verts[:, 1]
        z = verts[:, 2]

        vertices = np.vstack((x, y, z)).T
        tri_points = vertices[triangles]

        # extract the lists of x, y, z coordinates of the triangle
        # vertices and connect them by a "line" by adding None
        # this is a plotly convention for plotting segments
        Xe = []
        Ye = []
        Ze = []
        for T in tri_points:
            Xe.extend([T[k % 3][0] for k in range(4)] + [None])
            Ye.extend([T[k % 3][1] for k in range(4)] + [None])
            Ze.extend([T[k % 3][2] for k in range(4)] + [None])

        # return Xe, Ye, Ze
        wireframe = go.Scatter3d(
            x=Xe,
            y=Ye,
            z=Ze,
            mode="lines",
            name="wireframe",
            line=dict(color="rgb(70,70,70)", width=1),
        )
        return wireframe

    @classmethod
    def create_landmarks_plot(
        cls, landmark_names: List[str], verts: np.ndarray
    ) -> List[plotly.graph_objs.Scatter3d]:
        plots = []

        landmark_colors = dict(
            zip(cls.landmarks.keys(), px.colors.qualitative.Alphabet)
        )

        for lm_name in landmark_names:
            if lm_name not in cls.landmarks.keys():
                print(f"Landmark {lm_name} is not defined.")
                pass

            lm_index = cls.landmarks[lm_name]
            if isinstance(lm_index, tuple):
                lm = (verts[lm_index[0]] + verts[lm_index[1]]) / 2
            else:
                lm = verts[lm_index]

            plot = go.Scatter3d(
                x=[lm[0]],
                y=[lm[1]],
                z=[lm[2]],
                mode="markers",
                marker=dict(
                    size=8,
                    color=landmark_colors[lm_name],
                    opacity=1,
                ),
                name=lm_name,
            )

            plots.append(plot)

        return plots

    @classmethod
    def create_measurement_length_plot(
        cls, measurement_name: str, verts: np.ndarray, measurements, color: str
    ):
        measurement_landmarks_inds = cls.length_definitions[measurement_name]

        segments = {"x": [], "y": [], "z": []}
        for i in range(2):
            if isinstance(measurement_landmarks_inds[i], tuple):
                lm_tnp = (
                    verts[measurement_landmarks_inds[i][0]]
                    + verts[measurement_landmarks_inds[i][1]]
                ) / 2
            else:
                lm_tnp = verts[measurement_landmarks_inds[i]]
            segments["x"].append(lm_tnp[0])
            segments["y"].append(lm_tnp[1])
            segments["z"].append(lm_tnp[2])
        for ax in ["x", "y", "z"]:
            segments[ax].append(None)

        if measurement_name in measurements:
            m_viz_name = f"{measurement_name}: {measurements[measurement_name]:.2f}cm"
        else:
            m_viz_name = measurement_name

        return go.Scatter3d(
            x=segments["x"],
            y=segments["y"],
            z=segments["z"],
            marker=dict(
                size=4,
                color="rgba(0,0,0,0)",
            ),
            line=dict(color=color, width=10),
            name=m_viz_name,
        )

    @classmethod
    def create_measurement_circumference_plot(
        cls, measurement_name: str, verts: np.ndarray, joints, measurements, color: str
    ):
        """
        Create circumference measurement plot
        :param measurement_name: str, measurement name to plot
        :param verts: np.array (N,3) of vertices
        :param faces: np.array (F,3) of faces connecting the vertices
        :param color: str of color to color the measurement

        Return
        plotly object to plot
        """

        circumf_landmarks = cls.circumf_definitions[measurement_name]["LANDMARKS"]
        circumf_landmark_indices = [
            cls.landmarks[l_name] for l_name in circumf_landmarks
        ]
        circumf_n1, circumf_n2 = cls.circumf_definitions[measurement_name]["JOINTS"]
        circumf_n1, circumf_n2 = JOINT2IND[circumf_n1], JOINT2IND[circumf_n2]

        plane_origin = np.mean(verts[circumf_landmark_indices, :], axis=0)
        plane_normal = joints[circumf_n1, :] - joints[circumf_n2, :]

        mesh = trimesh.Trimesh(vertices=verts, faces=cls.faces)

        slice_segments, sliced_faces = trimesh.intersections.mesh_plane(
            mesh,
            plane_normal=plane_normal,
            plane_origin=plane_origin,
            return_faces=True,
        )  # (N, 2, 3), (N,)

        slice_segments = cls._filter_body_part_slices(
            slice_segments, sliced_faces, measurement_name
        )

        slice_segments_hull = cls._circumf_convex_hull(slice_segments)

        draw_segments = {"x": [], "y": [], "z": []}
        map_ax = {0: "x", 1: "y", 2: "z"}

        for i in range(slice_segments_hull.shape[0]):
            for j in range(3):
                draw_segments[map_ax[j]].append(slice_segments_hull[i, 0, j])
                draw_segments[map_ax[j]].append(slice_segments_hull[i, 1, j])
                draw_segments[map_ax[j]].append(None)

        if measurement_name in measurements:
            m_viz_name = f"{measurement_name}: {measurements[measurement_name]:.2f}cm"
        else:
            m_viz_name = measurement_name

        return go.Scatter3d(
            x=draw_segments["x"],
            y=draw_segments["y"],
            z=draw_segments["z"],
            mode="lines",
            line=dict(color=color, width=10),
            name=m_viz_name,
        )

    @classmethod
    def visualize(cls, verts, joints, measurements, title="Measurement visualization"):
        verts = verts.detach().cpu().numpy()
        joints = joints.detach().cpu().numpy()

        measurement_names = cls.measurement_names

        landmark_names = list(cls.landmarks.keys())

        # visualize model mesh
        fig = go.Figure()
        mesh_plot = cls.create_mesh_plot(verts)
        fig.add_trace(mesh_plot)

        # visualize joints
        joint_plot = cls.create_joint_plot(joints)
        fig.add_trace(joint_plot)

        # visualize wireframe
        wireframe_plot = cls.create_wireframe_plot(verts)
        fig.add_trace(wireframe_plot)

        # visualize landmarks
        if "LANDMARKS" in cls.cached_visualizations.keys():
            fig.add_traces(list(cls.cached_visualizations["LANDMARKS"].values()))
        else:
            landmarks_plot = cls.create_landmarks_plot(landmark_names, verts)
            cls.cached_visualizations["LANDMARKS"] = {
                landmark_names[i]: landmarks_plot[i] for i in range(len(landmark_names))
            }
            fig.add_traces(landmarks_plot)

        # visualize measurements
        measurement_colors = dict(
            zip(cls.measurement_types.keys(), px.colors.qualitative.Alphabet)
        )

        for m_name in measurement_names:
            if m_name not in cls.measurement_types.keys():
                print(f"Measurement {m_name} not defined.")
                pass

            if cls.measurement_types[m_name] == MeasurementType().LENGTH:
                if m_name in cls.cached_visualizations["LENGTHS"]:
                    measurement_plot = cls.cached_visualizations["LENGTHS"][m_name]
                else:
                    measurement_plot = cls.create_measurement_length_plot(
                        measurement_name=m_name,
                        verts=verts,
                        measurements=measurements,
                        color=measurement_colors[m_name],
                    )
                    cls.cached_visualizations["LENGTHS"][m_name] = measurement_plot

            elif cls.measurement_types[m_name] == MeasurementType().CIRCUMFERENCE:
                if m_name in cls.cached_visualizations["CIRCUMFERENCES"]:
                    measurement_plot = cls.cached_visualizations["CIRCUMFERENCES"][
                        m_name
                    ]
                else:
                    measurement_plot = cls.create_measurement_circumference_plot(
                        measurement_name=m_name,
                        verts=verts,
                        joints=joints,
                        measurements=measurements,
                        color=measurement_colors[m_name],
                    )
                    cls.cached_visualizations["CIRCUMFERENCES"][
                        m_name
                    ] = measurement_plot

            fig.add_trace(measurement_plot)

        fig.update_layout(
            scene_aspectmode="data",
            width=1000,
            height=700,
            title=title,
        )

        fig.show()


# if __name__ == "__main__":
# betas = np.zeros((1, 10))

# aug = AugmentBetasCam()

# betas_aug, front_target_pose_rotmats, front_target_glob_rotmats, side_target_pose_rotmats, side_target_glob_rotmats = \
#     aug.aug_betas(betas)
# target_smpl_output = aug.smpl_model(body_pose=front_target_pose_rotmats,
#                             global_orient=front_target_glob_rotmats,
#                             betas=betas_aug,
#                             pose2rot=False)
# # print(target_smpl_output.joints.shape)
# measurements = MeasureVerts.verts2meas(target_smpl_output.vertices[0], target_smpl_output.joints[0])
# MeasureVerts.visualize(target_smpl_output.vertices[0], target_smpl_output.joints[0], measurements)
# print(measurements)
