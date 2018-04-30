"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import colorsys


def get_n_different_colors(nr_of_colors):
    rgb_tuples = [colorsys.hsv_to_rgb(x * 1.0 / nr_of_colors, 0.5, 0.5) for x in range(nr_of_colors)]
    return [(r*255, g*255, b*255) for (r, g, b) in rgb_tuples]
