# Copyright (c) 2018 Foundry.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

import SocketServer
import argparse
import os
import importlib
import socket #to get machine hostname
import traceback

import numpy as np

from message_pb2 import *

# import libcaffe2_detectron_ops_gpu.so once and for all
import detectron.utils.c2 as c2_utils
c2_utils.import_detectron_ops() 

class MLTCPServer(SocketServer.TCPServer):
    def __init__(self, server_address, handler_class, auto_bind=True):
        self.verbose = True
        # Each directory in models/ containing a model.py file is an available ML model
        self.available_models = [name for name in next(os.walk('models'))[1]
            if os.path.isfile(os.path.join('models', name, 'model.py'))]
        self.models = {}
        for model in self.available_models:
            print('Importing models.{}.model'.format(model))
            self.models[model] = importlib.import_module('models.{}.model'.format(model)).Model()
        SocketServer.TCPServer.__init__(self, server_address, handler_class, auto_bind)
        return

class ImageProcessTCPHandler(SocketServer.BaseRequestHandler):
    """This request handler is instantiated once per connection."""

    def handle(self):
        # Read the data headers
        data_hdr = self.request.recv(12)
        sz = int(data_hdr)
        self.vprint('Receiving message of size: {}'.format(sz))

        # Read data
        data = self.recvall(sz)
        self.vprint('{} bytes read'.format(len(data)))

        # Parse the message
        req_msg = RequestWrapper()
        req_msg.ParseFromString(data)
        self.vprint('Message parsed')

        # Process message
        resp_msg = self.process_message(req_msg)

        # Serialize response
        self.vprint('Serializing message')
        s = resp_msg.SerializeToString()
        msg_len = resp_msg.ByteSize()
        totallen = 12 + msg_len
        msg = str(totallen).zfill(12) + s
        self.vprint('Sending response message of size: {}'.format(totallen))
        # self.request.sendall(msg)
        self.sendmsg(msg, totallen)
        self.vprint('-----------------------------------------------')

    def process_message(self, message):
        if message.HasField('r1'):
            self.vprint('Received info request')
            return self.process_info(message)
        elif message.HasField('r2'):
            self.vprint('Received inference request')
            return self.process_inference(message)
        else:
            # Pass error message to the client
            return self.errormsg("Server received unindentified request from client.")            

    def process_info(self, message):
        resp_msg = RespondWrapper()
        resp_msg.info = True
        resp_info = RespondInfo()
        resp_info.num_models = len(self.server.available_models)
        # Add all model info into the message
        for model in self.server.available_models:
            m = resp_info.models.add()
            m.name = model
            m.label = self.server.models[model].get_name()
            # Add inputs
            for inp_name, inp_channels in self.server.models[model].get_inputs().items():
                inp = m.inputs.add()
                inp.name = inp_name
                inp.channels = inp_channels
            # Add outputs
            for out_name, out_channels in self.server.models[model].get_outputs().items():
                out = m.outputs.add()
                out.name = out_name
                out.channels = out_channels
            # Add options
            for opt_name, opt_value in self.server.models[model].get_options().items():
                if type(opt_value) == int:
                    opt = m.int_options.add()
                elif type(opt_value) == float:
                    opt = m.float_options.add()
                elif type(opt_value) == bool:
                    opt = m.bool_options.add()
                elif type(opt_value) == str:
                    opt = m.string_options.add()
                    # TODO: Implement multiple choice
                else:
                    # Send an error response message to the Nuke Client
                    option_error = ("Model option of type {} is not implemented. "
                        "Broadcasted options need to be one of bool, int, float, str."
                    ).format(type(opt_value))
                    return self.errormsg(option_error)
                opt.name = opt_name
                opt.values.extend([opt_value])
            # Add buttons
            for button_name, button_value in self.server.models[model].get_buttons().items():
                if type (button_value) == bool:
                    button = m.button_options.add()
                else:
                    return self.errormsg("Model button needs to be of type bool.")
                button.name = button_name
                button.values.extend([button_value])

        # Add RespondInfo message to RespondWrapper
        resp_msg.r1.CopyFrom(resp_info)

        return resp_msg

    def process_inference(self, message):
        req = message.r2
        m = req.model
        self.vprint('Requesting inference on model: {}'.format(m.name))

        # Parse model options
        opt = {}
        for options in [m.bool_options, m.int_options, m.float_options, m.string_options]:
            for option in options:
                opt[option.name] = option.values[0]
        # Set model options
        self.server.models[m.name].set_options(opt)
        # Parse model buttons
        btn = {}
        for button in m.button_options:
            btn[button.name] = button.values[0]       
        self.server.models[m.name].set_buttons(btn)

        # Parse images
        img_list = []
        for byte_img in req.images:
            img = np.fromstring(byte_img.image, dtype='<f4')     
            height = byte_img.height
            width = byte_img.width
            channels = byte_img.channels
            img = np.reshape(img, (channels, height, width))
            img = np.transpose(img, (1, 2, 0))
            img = np.flipud(img)
            img_list.append(img)
        try:
            # Running inference
            self.vprint('Starting inference')
            res = self.server.models[m.name].inference(img_list)
            # Creating response messsage
            resp_msg = RespondWrapper()
            resp_msg.info = True
            resp_inf = RespondInference()
            num_images = 0
            num_objects = 0
            for obj in res:
                # Send an image back to Nuke
                if isinstance(obj, np.ndarray):
                    num_images += 1
                    img = np.flipud(obj)
                    image = resp_inf.images.add()
                    image.width = np.shape(img)[1]
                    image.height = np.shape(img)[0]
                    image.channels = np.shape(img)[2]
                    img = np.transpose(img, (2, 0, 1))
                    image.image = img.tobytes()
                # Send a general object back to Nuke
                elif isinstance(obj, FieldValuePairAttrib):
                    num_objects += 1
                    resp_inf.objects.extend([obj])
                else:
                    exception_msg = ("Object returned from model inference is of type {}."
                        "It should be an np.array image or a general FieldValuePairAttrib".format(type(obj)))
                    raise Exception(exception_msg)
            resp_inf.num_images = num_images
            resp_inf.num_objects = num_objects
            self.vprint('Infering back {} image(s) and {} object(s)'.format(num_images, num_objects))
            if num_images == 0 and num_objects == 0:
                raise Exception("No images or non-image objects were returned from model inference")
            # Add RespondInference message to RespondWrapper
            resp_msg.r2.CopyFrom(resp_inf)
        except Exception as e:
            # Pass error message to the client
            self.vprint('Exception caught on inference on model:')
            self.vprint(str(traceback.print_exc()))
            resp_msg = self.errormsg(str(e))
            
        return resp_msg

    def recvall(self, n):
        """Helper function to receive n bytes or return None if EOF is hit"""
        data = b''
        while len(data) < n:
            packet = self.request.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def sendmsg(self, msg, msglen):
        totalsent = 0
        while totalsent < msglen:
            sent = self.request.send(msg[totalsent:])
            if sent == 0:
                raise RuntimeError("Socket connection broken")
            totalsent = totalsent + sent

    def errormsg(self, error):
        """Create an error message to send a Server error to the Nuke Client"""
        resp_msg = RespondWrapper()
        resp_msg.info = True
        error_msg = Error() # from message_pb2.py
        error_msg.msg = error
        resp_msg.error.CopyFrom(error_msg)
        return resp_msg

    def vprint(self, string):
        if self.server.verbose:
            print('Server -> ' + string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Machine Learning inference server.')
    parser.add_argument('port', type=int, help='Port number for the server to listen to.')
    args = parser.parse_args()

    # Get the current hostname of the server
    server_hostname = socket.gethostbyname(socket.gethostname())
    # Create the server
    server = MLTCPServer((server_hostname, args.port), ImageProcessTCPHandler, False)

    # Bind and activate the server
    server.allow_reuse_address = True 
    server.server_bind()     
    server.server_activate()
    print('Server -> Listening on port: {}'.format(args.port))
    server.serve_forever()