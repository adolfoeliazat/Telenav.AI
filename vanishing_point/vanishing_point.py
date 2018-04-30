"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import math
import numpy as np
from operator import methodcaller

import cv2
from sklearn.cluster import DBSCAN

from apollo_python_common.geometry.line_segment import LineSegment
from apollo_python_common.lightweight_types import Point


class VanishingPointDetector(object):
    # thresholds for almost vertical
    LIMIT1 = math.tan(86 * np.pi / 180)
    LIMIT2 = math.tan(-86 * np.pi / 180)
    # thresholds for almost horizontal
    LIMIT3 = math.tan(5 * np.pi / 180)
    LIMIT4 = math.tan(-5 * np.pi / 180)
    # angle threshold for 2 lines to be considered parallel
    ANGLE_THRESHOLD = math.cos(3 * np.pi / 180)

    def __init__(self, image_width_reference_size=640, nr_used_lines=20):
        '''
        Constructor for the vanishing point detector
        :param image_width_reference_size: image width for scale at which computations should be done
        :param nr_used_lines: number of lines used in computing the vanishing point
        '''
        self.image_width_reference_size = image_width_reference_size
        self.nr_used_lines = nr_used_lines
        self.gray_image = None
        self.resized_gray_image = None
        self.edges_image = None

    def __get_ppht_detections(self, binary_image):
        rho_resolution = 1
        theta_resolution = np.pi / 180
        bin_threshold = int(binary_image.shape[1] / 25)
        min_line_length = int(bin_threshold / 2)
        max_line_gap = 4
        lines = cv2.HoughLinesP(binary_image, rho_resolution, theta_resolution,
                                bin_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
        if lines is None:
            return []
        segment_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            segment_lines.append(LineSegment(Point(x1, y1), Point(x2, y2)))
        return segment_lines

    def __is_valid_orientation(self, line):
        slope = line.slope()
        if slope == None:
            return False
        return (slope < self.LIMIT1 and slope > self.LIMIT3) or (slope < self.LIMIT4 and slope > self.LIMIT2)

    def __filter_lines(self, lines):
        filtered_lines = [line for line in lines if self.__is_valid_orientation(line)]
        return filtered_lines

    def __get_clusters(self, intersections, region_width):
        if len(intersections) == 0:
            return []
        distance_threshold = region_width / 80
        db = DBSCAN(distance_threshold, 1).fit(intersections)
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        intersections = np.array(intersections)
        clusters = [intersections[labels == i] for i in range(n_clusters_)]
        clusters.sort(key=len)
        return clusters

    def __compute_best_solution(self, lines, region_width):
        intersections = []
        for i in range(len(lines) - 1):
            for j in range(i+1, len(lines)):
                if not lines[i].is_parallel_with_line(lines[j], self.ANGLE_THRESHOLD):
                    intersection = lines[i].get_intersection_with_line(lines[j])
                    if intersection is not None:
                        intersections.append(intersection)
        clusters = self.__get_clusters(intersections, region_width)
        if len(clusters) > 0:
            # compute the vanishing point by getting an average center of the cluster
            largest_cluster = clusters[-1]
            sum_x, sum_y = [sum(idx) for idx in zip(*largest_cluster)]
            solution_cluster_size = len(largest_cluster)
            solution_x = round(sum_x / solution_cluster_size)
            solution_y = round(sum_y / solution_cluster_size)
            return Point(solution_x, solution_y), solution_cluster_size
        return None, 0

    def get_vanishing_point(self, frame):
        '''
        Computes vanishing point and its confidence for the input image
        :param frame: BGR or Grayscale image
        :return: vanishing point and confidence
        '''
        frame_width = int(frame.shape[1])
        frame_height = int(frame.shape[0])
        row_limit = int(frame_height / 2)
        needed_roi = frame[row_limit:frame_height, 0:frame_width]

        if len(frame.shape) > 2:
            self.gray_image = cv2.cvtColor(needed_roi, cv2.COLOR_BGR2GRAY)
        else:
            self.gray_image = needed_roi

        resize_factor = min(self.image_width_reference_size / frame_width, 1.0)
        if resize_factor < 0.9:
            self.resized_gray_image = cv2.resize(self.gray_image, None, fx =resize_factor, fy = resize_factor, interpolation=cv2.INTER_AREA)
        else:
            self.resized_gray_image = self.gray_image

        otsu_threshold, thresholded_img = cv2.threshold(self.resized_gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_threshold = otsu_threshold * 0.5
        high_threshold = otsu_threshold
        self.edges_image = cv2.Canny(self.resized_gray_image, low_threshold, high_threshold, apertureSize=3, L2gradient=False)

        ppht_lines = self.__get_ppht_detections(self.edges_image)
        filtered_lines = self.__filter_lines(ppht_lines)
        if len(filtered_lines) == 0:
            return None, 0

        filtered_lines = sorted(filtered_lines, key=methodcaller('approx_length'), reverse=True)
        filtered_used_lines = filtered_lines[:self.nr_used_lines]

        resized_frame_width = self.resized_gray_image.shape[1]
        vp_result, solution_cluster_size = self.__compute_best_solution(filtered_used_lines, resized_frame_width)

        if vp_result is None:
            return None, 0

        # 90% of the lines need to intersect in the same region to have total confidence in the vp
        nr_intersecting_lines_needed = round(self.nr_used_lines * 0.9)
        nr_intersections_needed = nr_intersecting_lines_needed * (nr_intersecting_lines_needed - 1) / 2
        confidence = min(solution_cluster_size / nr_intersections_needed, 1.0)

        vp_result_x = vp_result.x
        vp_result_y = vp_result.y
        if resize_factor < 0.9:
            vp_result_x = round(vp_result_x / resize_factor)
            vp_result_y = round(vp_result_y / resize_factor)
        vp_result_y += row_limit
        return Point(int(vp_result_x), int(vp_result_y)), confidence
