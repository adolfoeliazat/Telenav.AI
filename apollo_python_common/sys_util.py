"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
import os
import time
import signal

DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_SEC = 30


def graceful_shutdown(logger, timeout_sec=DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_SEC):
    '''
    Shutdown the current process, allowing the workers (if they exists) to complete their works before.
    '''
    if timeout_sec < 0:
        logger.info("Invalid timeout for graceful shutdown. Setting it to default value {}".format(DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_SEC))
        timeout_sec = DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT_SEC
    for sec in reversed(range(timeout_sec)):
        logger.info("The app will stop in {} seconds".format(sec))
        time.sleep(1)
    # Here we assume that the graceful_shutdown method can be called from a thread which is different than main thread.
    os.kill(os.getpid(), signal.SIGINT)  # This works only on Linux based systems.
