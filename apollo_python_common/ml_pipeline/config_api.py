"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import os


class MQ_Param:
    COMPONENT = "component"
    ALGORITHM = "algorithm"
    MQ_HOST = "mq_host"
    MQ_PORT = "mq_port"
    MQ_USERNAME = "mq_username"
    MQ_PASSWORD = "mq_password"
    MQ_INPUT_QUEUE_NAME = "mq_input_queue_name"
    MQ_INPUT_ERRORS_QUEUE_NAME = "mq_input_errors_queue_name"
    MQ_OUTPUT_QUEUE_NAME = "mq_output_queue_name"
    MAX_INTERNAL_QUEUE_SIZE = "max_internal_queue_size"
    PREDICT_BATCH_SIZE = "predict_batch_size"
    ELASTICSEARCH_HOST = "elasticsearch_host"
    ELASTICSEARCH_AUDIT_INDEX_NAME = "elasticsearch_audit_index_name"
    NO_ACK = "no_ack"
    MQ_PREFETCH_COUNT = "mq_prefetch_count"
    MQ_NR_PREPROCESS_THREADS = "nr_preprocess_threads"
    MQ_NR_PREDICT_THREADS = "nr_predict_threads"


def get_config_param(key, config_dict, default_value=None):
    if key in os.environ:
        return os.environ.get(key)

    if key in config_dict:
        return config_dict[key]

    return default_value
