"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import argparse
import grpc
from apollo_python_common.protobuf import inference_service_pb2_grpc
from apollo_python_common.protobuf import inference_service_pb2
from apollo_python_common.proto_api import serialize_metadata
import apollo_python_common.log_util as log_util
import logging
import apollo_python_common.io_utils as io_utils


def start_client(host, port, input_path, output_path):
    io_utils.create_folder(output_path)
    channel = grpc.insecure_channel(host + ":" + str(port))
    stub = inference_service_pb2_grpc.InferenceServiceStub(channel)
    metadata = stub.process(inference_service_pb2.DetectionRequest(
        images_path=input_path))
    if metadata is not None:
        serialize_metadata(metadata, output_path, 'rois')
    else:
        logger.error('An error has occurred processing folder. See server log for more details.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path", type=str, required=True)
    parser.add_argument(
        "-o", "--output_path", type=str, required=False, default="./out")
    parser.add_argument(
        "-t", "--host", type=str, required=False, default="localhost")
    parser.add_argument(
        "-p", "--port", type=int, required=False, default=31338)
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    host = args.host
    port = args.port
    start_client(host, port, input_path, output_path)


if __name__ == "__main__":
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    logger.info("Processing with GRPC client...")
    try:
        main()
    except Exception as err:
        logger.error(err.message, exc_info=True)
    logger.info("GRPC client finalised.")
