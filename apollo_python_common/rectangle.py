"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
from math import sqrt
import numpy as np


class Rectangle:

    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def get_intersection_area(self, sec_rect):
        rect_a = self
        dx = min(rect_a.xmax, sec_rect.xmax) - max(rect_a.xmin, sec_rect.xmin)
        dy = min(rect_a.ymax, sec_rect.ymax) - max(rect_a.ymin, sec_rect.ymin)
        if (dx >= 0) and (dy >= 0):
            return float(dx * dy)
        else:
            return float(0)

    def area(self):
        return self.width() * self.height()

    def width(self):
        return float(self.xmax - self.xmin)

    def height(self):
        return float(self.ymax - self.ymin)

    def get_overlapped_rect(self, sec_rect):
        rect_a = self
        x_min = max(rect_a.xmin, sec_rect.xmin)
        y_min = max(rect_a.ymin, sec_rect.ymin)
        x_max = min(rect_a.xmax, sec_rect.xmax)
        y_max = min(rect_a.ymax, sec_rect.ymax)
        if x_min <= x_max and y_min <= y_max:
            return Rectangle(x_min, y_min, x_max, y_max)
        else:
            return Rectangle(0, 0, 0, 0)

    def get_distance_between_centers(self, sec_rect):
        center_1_x = (self.xmin + self.xmax) / 2
        center_1_y = (self.ymin + self.ymax) / 2
        center_2_x = (sec_rect.xmin + sec_rect.xmax) / 2
        center_2_y = (sec_rect.ymin + sec_rect.ymax) / 2
        dist = sqrt((center_2_x - center_1_x)**2 + (center_2_y - center_1_y)**2)
        return dist

    def get_bounding_box_rect(self, sec_rect):
        first_rect = self
        return Rectangle(min(first_rect.xmin, sec_rect.xmin),
                         min(first_rect.ymin, sec_rect.ymin),
                         max(first_rect.xmax, sec_rect.xmax),
                         max(first_rect.ymax, sec_rect.ymax))

    def get_average_box_rect(self, sec_rect):
        first_rect = self
        return Rectangle(np.mean([first_rect.xmin, sec_rect.xmin]),
                         np.mean([first_rect.ymin, sec_rect.ymin]),
                         np.mean([first_rect.xmax, sec_rect.xmax]),
                         np.mean([first_rect.ymax, sec_rect.ymax]))

    def intersection_over_union(self, sec_rec):
        intersect_area = self.get_intersection_area(sec_rec)
        union_area = self.area() + sec_rec.area() - intersect_area
        if union_area != 0:
            ret_val = intersect_area / union_area
        else:
            ret_val = 0
        return float(ret_val)

    def __repr__(self):
        return 'xmin {}, ymin {}, xmax {}, ymax {}'.format(self.xmin, self.ymin, self.xmax, self.ymax)

