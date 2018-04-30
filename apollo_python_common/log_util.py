"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import logging
import logging.config
import os


def config(log_file_name):
    log_file_name = os.path.abspath(log_file_name)
    full_path_log_file = os.path.abspath(os.path.join(os.path.split(log_file_name)[0].split('python_modules')[0], "python_modules","apollo_python_common","logs", "logging.ini"))
    logging.config.fileConfig(full_path_log_file,
                              defaults={'logfilename': os.path.splitext(os.path.basename(log_file_name))[0] + '.log'})
