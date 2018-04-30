"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
"""
Download tool for resources on ftp
"""

import argparse

import apollo_python_common.ftp_utils as ftp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ftp_file_path",
                        type=str, required=True)
    parser.add_argument("-o", "--local_file_path",
                        type=str, required=True)
    parser.add_argument("-f", "--ftp_server",
                        type=str, required=False, default="localhost")
    parser.add_argument("-u", "--ftp_username",
                        type=str, required=False, default="user")
    parser.add_argument("-p", "--ftp_password",
                        type=str, required=False, default="password")
    args = parser.parse_args()
    ftp.file_copy_ftp_to_local( args.ftp_server, args.ftp_username, args.ftp_password, args.ftp_file_path, args.local_file_path )


if __name__ == "__main__":
    main()
