"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import configparser

# TODO: make singleton
class Configuration(object):
    """
    Configuration class used to hold values used all over the project
    """

    def __init__(self, configuration_path):
        self.configuration_path = configuration_path
        # see ml.cfg for more info
        self.min_size = 0
        self.ratio = 0
        self.epsilon = 0
        self.image_size = (0, 0)
        self.split_ratio = 0
        self.invalid_id = 0
        # type     - id as defined in the protobuf model
        # class_id - starting from 1..(number of sections defined in ml.cfg)
        #            used as an id in indexed PNG
        # see https://en.wikipedia.org/wiki/Indexed_color
        self.type2class_id = {}
        self.class_id2type = {}
        # map between the protobuf id and the section name in ml.cfg
        # TODO: perhaps use the protobuf associated name
        self.type2name = {}
        # Useful for tensorflow
        self.name2class_id = {}
        # each class_id has an associated color in indexed PNG label files
        self.palette = [0, 0, 0]
        self.class_id2threshold = {}
        self.algorithm = ""
        self.algorithm_version = ""

def load_configuration(configuration_path="./config/ml.cfg"):
    """
    Loads the configuration file and initializes the configuration class
    :param configuration_path: file path
    :return: Configuration object
    """

    parser = configparser.ConfigParser()
    parser.read(configuration_path)

    config = Configuration(configuration_path)
    config.min_size = int(parser.get('general', 'min_size'))
    config.ratio = float(parser.get('general', 'ratio'))
    config.epsilon = float(parser.get('general', 'epsilon'))
    config.image_size = (int(parser.get('general', 'width')),
                         int(parser.get('general', 'height')))
    config.split_ratio = float(parser.get('general', 'split_ratio'))
    config.invalid_id = int(parser.get('general', 'invalid_id'))
    config.algorithm = parser.get('general', 'algorithm')
    config.algorithm_version = parser.get('general', 'algorithm_version')

    class_id = 0
    for section in parser.sections():
        if section != 'general':
            type_id = int(parser.get(section, 'id'))
            threshold = float(parser.get(section, 'threshold'))
            color = parser.get(section, 'color').split(',')
            config.type2class_id[type_id] = class_id
            config.class_id2type[class_id] = type_id
            config.class_id2threshold[class_id] = threshold
            config.type2name[type_id] = section
            config.name2class_id[section] = class_id
            config.palette.extend([int(c) for c in color])
            class_id += 1

    return config
