"""
Copyright 2018-2019 Telenav (http://telenav.com)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

class MessageEnvelope:
    '''
    Envelope for message queue messages received
    '''
    def __init__(self, input_id, input_body, body, **kwargs):
        '''

        :param input_id: unique identifier in the message queue
        :param input_body: message as it was received from MQ
        :param body: current message body
        :param kwargs: helper optional arguments
        '''
        self.input_id = input_id
        self.input_body = input_body
        self.body = body
        self.args = kwargs
        self.processing_time = list()

    def get_with_new_body(self, body):
        msg = MessageEnvelope(self.input_id, self.input_body, body, **self.args)
        msg.processing_time = self.processing_time
        return msg
