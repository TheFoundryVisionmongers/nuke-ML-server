# Copyright (c) 2019 Alexander Mishurov.
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

import os

import cv2
import numpy as np
import tensorflow as tf
from skimage.transform import estimate_transform, warp
import message_pb2

from ..baseModel import BaseModel


DATA_PATH = './models/arcore/data'

FACEMESH_MODEL_PATH = os.path.join(
    DATA_PATH, 'facemesh-lite_nocrawl-2019_01_14-v0.tflite')

DETECTOR_MODEL_PATH = os.path.join(DATA_PATH, 'opencv_face_detector_uint8.pb')

DETECTOR_CONFIG_PATH = os.path.join(DATA_PATH, 'opencv_face_detector.pbtxt')

FACEMESH_OBJ_PATH = os.path.join(DATA_PATH, 'canonical_face_mesh.obj')


def read_obj(path):
    v_values = []
    vt_values = []
    vn_values = []
    f_indices = []

    def parse_values(line):
        return [float(v) for v in line.split()[1:]]

    def parse_indices(line):
        return [[int(i) - 1 for i in t.split("/")] for t in line.split()[1:]]

    # Map .obj lines to parsing functions
    parse_map = {
        "v ": [v_values, parse_values],
        "vt": [vt_values, parse_values],
        "vn": [vn_values, parse_values],
        "f ": [f_indices, parse_indices],
    }

    def parse_line(l):
        attr = l[:2]
        if attr in parse_map.keys():
            p = parse_map[attr]
            p[0].append(p[1](l))

    with open(path) as obj:
        [parse_line(l) for l in obj.readlines()]

    # Construct flat arrays of face indices and face corners' attributes
    uvs = []
    normals = []
    faces = []
    for f in f_indices:
        for indices in f:
            faces.append(indices[0])
            uv = vt_values[indices[1]][:]
            # Add 0 and 1 to every uv for Nuke
            uv.extend([0., 1.])
            uvs.extend(uv)
            norm = vn_values[indices[2]][:]
            # Invert normal's z for Nuke
            norm[2] *= -1
            normals.extend(norm)

    return faces, uvs, normals


class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__()
        self.name = 'Google ARCore'

        # Define options
        self.detect_face = True
        self.options = ('detect_face',)

        # Define inputs/outputs
        self.inputs = {'input': 3}
        self.outputs = {'output': 3}

        # Configure face detector and mesh models
        self.interpreter = tf.lite.Interpreter(model_path=FACEMESH_MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.detector = cv2.dnn.readNetFromTensorflow(
            DETECTOR_MODEL_PATH, DETECTOR_CONFIG_PATH)

        input_details = self.interpreter.get_input_details()
        self.input_shape = input_details[0]['shape']
        self.input_index = input_details[0]['index']
        output_details = self.interpreter.get_output_details()
        self.output_index = output_details[0]['index']

        # Define 3 points of the input for simiarity transformation
        self.dst_points = np.array([
            [0, 0], [0, self.input_shape[1] - 1], [self.input_shape[2] - 1, 0]
        ])

        # Load canonical face geometry data
        self.faces, self.uvs, self.normals = read_obj(FACEMESH_OBJ_PATH)

    def vprint(self, msg):
        print("{} -> {}".format(self.name, msg))

    def inference(self, image_list):
        image = image_list[0]
        image_fp32 = self.linear_to_srgb(image)
        image = (image_fp32 * 255).astype(np.uint8)

        h, w = image.shape[:2]
        box = self.predict_face_box(image) if self.detect_face else (0, 0, w, h)
        if box is None:
            # It would be nice to send error messages to a client
            script = "nuke.error('No faces found')\n"
            script_msg = message_pb2.FieldValuePairAttrib()
            script_msg.name = "PythonScript"
            script_msg_val = script_msg.values.add()
            script_msg_str = script_msg_val.string_attributes.add()
            script_msg_str.values.extend([script])
            return [script_msg]

        # Draw rectangle around detected face
        (x, y, r, b) = box
        cv2.rectangle(image, (x, y), (r, b), (0, 255, 0), 2)

        # Predict points of the detected face
        positions = self.predict_face_points(image, box)

        # Create protobuf object
        geo = message_pb2.FieldValuePairAttrib()
        geo.name = "Geo"
        geo_val = geo.values.add()

        points = geo_val.float_attributes.add()
        points.name = "points"
        points.values.extend(positions)

        attrs_map = {
            "indices": ["int_attributes", self.faces],
            "uv Group_Vertices": ["float_attributes", self.uvs],
            "N Group_Vertices": ["float_attributes", self.normals],
        }

        for k, v in attrs_map.items():
            attr = getattr(geo_val, v[0]).add()
            attr.name = k
            attr.values.extend(v[1])

        # Convert image for nuke and with geometry so the model can
        # work both the image processing and 3d nodes
        image = image.astype(np.float32) / 255.
        image = self.srgb_to_linear(image)
        return [image, geo]

    def predict_face_box(self, image):
        self.vprint("Predicting face bounding box...")

        h, w = image.shape[:2]
        # Create an input tensor for face detection
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300), (104, 177, 123))
        self.detector.setInput(blob)

        # Run network
        detections = self.detector.forward()

        # Get first plausible detection
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if (confidence < 0.9):
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = box.astype("int")

            self.vprint("Found face baounding box {}, {}, {}, {}".format(*box))
            return box.astype("int")

        self.vprint("No faces found")

    def transform_points(self, points, tform, image_height):
        # Extract z and apply inverse trasformation
        points = np.reshape(points, [-1, 3]).T
        z = points[2, :].copy() / tform.params[0, 0]
        # invert and offset z in order for points to correspond Nuke's space
        # and were in front of the XY plane
        z *= -1
        min_z = z.min()
        z -= min_z
        # Apply the inverse trasformation for x and y so the points in 3d space
        # would correspond to the points on the image
        points[2, :] = 1
        points = np.dot(np.linalg.inv(tform.params), points)
        # Invert y
        points[1, :] = image_height - points[1, :]
        points = np.vstack((points[:2, :], z))
        return points.T.flatten()

    def predict_face_points(self, image, box):
        self.vprint("Predicting face points...")

        (x, y, r, b) = box
        w = r - x
        h = b - y

        # Define 3 points of the detected rectangle for simiarity transformation
        center = np.array([r - w / 2.0, b - h / 2.0])
        size = int(((w + h) / 2) * 1.6)
        src_points = np.array([
            [center[0] - size / 2, center[1] - size / 2],
            [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2]
        ])

        # Get a similarity transform
        tform = estimate_transform('similarity', src_points, self.dst_points)

        # Create an input tensor for point prediction
        input_image = warp(
            image, tform.inverse, output_shape=self.input_shape[1:3])
        input_image = np.asarray(input_image).astype(np.float32)
        input_image = np.reshape(input_image, self.input_shape)

        # Run network
        self.interpreter.set_tensor(self.input_index, input_image)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_index)
        return self.transform_points(
            output_data.flatten(), tform, image.shape[0])
