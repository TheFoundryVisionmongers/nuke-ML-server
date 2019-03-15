import SocketServer
import argparse
import os
import importlib
import socket #to get machine hostname

import numpy as np

from message_pb2 import *

# import libcaffe2_detectron_ops_gpu.so once and for all
import detectron.utils.c2 as c2_utils
c2_utils.import_detectron_ops() 

class DLTCPServer(SocketServer.TCPServer):
    def __init__(self, server_address, handler_class, auto_bind=True):
        self.verbose = True
        self.available_models = next(os.walk('models'))[1]
        self.models = {}
        for model in self.available_models:
            self.models[model] = importlib.import_module('models.{}.model'.format(model)).Model()

        SocketServer.TCPServer.__init__(self, server_address, handler_class, auto_bind)
        return

class ImageProcessTCPHandler(SocketServer.BaseRequestHandler):
    """
    This request handler is instantiated once per connection.
    """

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

        # Serializing response
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
            self.vprint('Received unidentified request')
            # TODO: Implement error handling

    def process_info(self, message):
        resp_msg = RespondWrapper()
        resp_msg.info = True
        resp_info = RespondInfo()
        resp_info.numModels = len(self.server.available_models)
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
                    opt = m.intOptions.add()
                elif type(opt_value) == float:
                    opt = m.floatOptions.add()
                elif type(opt_value) == bool:
                    opt = m.boolOptions.add()
                elif type(opt_value) == str:
                    opt = m.stringOptions.add()
                    # TODO: Implement multiple choice
                else:
                    raise NotImplementedError
                    # TODO: Do better error handling
                opt.name = opt_name
                opt.value = opt_value

        # Add RespondInfo message to RespondWrapper
        resp_msg.r1.CopyFrom(resp_info)

        return resp_msg

    def process_inference(self, message):
        req = message.r2
        m = req.model
        self.vprint('Requesting inference on model: {}'.format(m.name))

        # Parse model options
        opt = {}
        
        for options in [m.boolOptions, m.intOptions, m.floatOptions, m.stringOptions]:
            for option in options:
                opt[option.name] = option.value

        # Set model options
        self.server.models[m.name].set_options(opt)

        # Parse images
        img_list = []
        for byte_img in req.image:
            img = np.fromstring(byte_img.image, dtype='<f4')     
            height = byte_img.height
            width = byte_img.width
            channels = byte_img.channels
            img = np.reshape(img, (channels, height, width))
            img = np.transpose(img, (1, 2, 0))
            img = np.flipud(img)
            # img = np.clip(img, 0, 1)
            # img *= 255
            img_list.append(img)

        # Running inference
        self.vprint('Starting inference')
        res = self.server.models[m.name].inference(img_list)

        # Creating response messsage
        resp_msg = RespondWrapper()
        resp_msg.info = True
        resp_inf = RespondInference()
        resp_inf.numImages = len(res)
        for img in res:
            # img = img.astype(np.float32) / 255.
            img = np.flipud(img)
            image = resp_inf.image.add()
            image.width = np.shape(img)[0]
            image.height = np.shape(img)[1]
            image.channels = np.shape(img)[2]
            img = np.transpose(img, (2, 0, 1))
            image.image = img.tobytes()

        # Add RespondInference message to RespondWrapper
        resp_msg.r2.CopyFrom(resp_inf)

        return resp_msg

    def recvall(self, n):
        # Helper function to recv n bytes or return None if EOF is hit
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

    def vprint(self, string):
        if self.server.verbose:
            print('Server -> ' + string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Learning inference server.')
    parser.add_argument('port', type=int, help='Port number for the server to listen to.')
    args = parser.parse_args()

    # Get the current hostname of the server
    server_hostname = socket.gethostbyname(socket.gethostname())
    # Create the server
    server = DLTCPServer((server_hostname, args.port), ImageProcessTCPHandler, False)

    # Bind and activate the server
    server.allow_reuse_address = True 
    server.server_bind()     
    server.server_activate()
    print('Server -> Listening on port: {}'.format(args.port))
    server.serve_forever()