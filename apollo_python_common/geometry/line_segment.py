"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import math

from apollo_python_common.lightweight_types import Point


class LineSegment(object):
    def __init__(self, point_a, point_b):
        '''
        Constructor that takes 2 Point namedtuples
        :param point_a: Point
        :param point_b: Point
        '''
        self.point_a = point_a
        self.point_b = point_b

    def length(self):
        return math.sqrt((self.point_a.x - self.point_b.x) ** 2 + (self.point_a.y - self.point_b.y) ** 2)

    def approx_length(self):
        return abs(self.point_a.x - self.point_b.x) + abs(self.point_a.y - self.point_b.y)

    def is_vertical(self):
        return self.point_a.x == self.point_b.x

    def is_horizontal(self):
        return self.point_a.y == self.point_b.y

    def slope(self):
        if not self.is_vertical():
            return (self.point_b.y - self.point_a.y) / (self.point_b.x - self.point_a.x)
        else:
            return None

    def get_intersection_with_line(self, other_line):
        d = (self.point_a.x - self.point_b.x) * (other_line.point_a.y - other_line.point_b.y) - \
            (self.point_a.y - self.point_b.y) * (other_line.point_a.x - other_line.point_b.x)

        if d == 0:
            return None

        pre = self.point_a.x * self.point_b.y - self.point_b.x * self.point_a.y
        post = other_line.point_a.x * other_line.point_b.y - other_line.point_b.x * other_line.point_a.y

        x = (pre * (other_line.point_a.x - other_line.point_b.x) - (self.point_a.x - self.point_b.x) * post) / d
        y = (pre * (other_line.point_a.y - other_line.point_b.y) - (self.point_a.y - self.point_b.y) * post) / d

        return Point(x, y)

    def is_parallel_with_line(self, other_line, epsilon):
        '''
        Check if lines are parallel within a epsilon limit : epsilon = math.cos(3 * np.pi / 180) will consider lines with a difference of 3 degrees (or less) as parallel lines
        :param other_line: line to check if is parallel with self
        :param epsilon: the angle threshold for 2 lines to be considered parallel
        :return: True if lines are parallel, False otherwise
        '''
        ux = self.point_b.x - self.point_a.x
        uy = self.point_b.y - self.point_a.y
        vx = other_line.point_b.x - other_line.point_a.x
        vy = other_line.point_b.y - other_line.point_a.y

        dot_product = ux * vx + uy * vy
        u_magnitude_square = ux ** 2 + uy ** 2
        v_magnitude_square = vx ** 2 + vy ** 2

        threshold = u_magnitude_square * v_magnitude_square * epsilon * epsilon
        return dot_product ** 2 > threshold