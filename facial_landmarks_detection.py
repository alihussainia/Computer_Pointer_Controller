import cv2
from openvino.inference_engine.ie_api import IENetwork, IECore
import pprint

# To crop the eyes from the face, we use a square sized with 1/5 the width of the face.
EYE_FACE_COEF = 0.2

class Facial_Landmarks_Detection:
    def __init__(self, model_name, device='CPU', extensions=None, perf_counts="False"):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extensions = extensions
        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        self.net = None
        self.pp = None
        if perf_counts == "True":
            self.pp = pprint.PrettyPrinter(indent=4)

    def load_model(self):
        core = IECore()
        if self.extensions != None:
            core.add_extension(self.extensions, self.device)
        self.net = core.load_network(network=self.model, device_name=self.device, num_requests=1)

    def predict(self, image):
        preprocessed_image = self.preprocess_input(image)
        output = self.net.infer({self.input_name: preprocessed_image})
        if self.pp is not None:
            self.pp.pprint(self.net.requests[0].get_perf_counts())
        return self.preprocess_output(next(iter(output.values()))[0], image)

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, outputs, image):
        width = int(image.shape[1])
        height = int(image.shape[0])
        eye_square_size = int(width * EYE_FACE_COEF)
        left_eye = cv2.getRectSubPix(image, (eye_square_size, eye_square_size), (outputs[0] * width + eye_square_size / 2, outputs[1] * height + eye_square_size / 2))
        right_eye = cv2.getRectSubPix(image, (eye_square_size, eye_square_size), (outputs[2] * width + eye_square_size / 2, outputs[3] * height + eye_square_size / 2))
        return left_eye, right_eye