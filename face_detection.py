import cv2
from openvino.inference_engine.ie_api import IECore, IENetwork
import pprint

# default threshold
THRESHOLD = 0.5

class Face_Detection:
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
        coords = self.preprocess_output(output[self.output_name])
        if not coords:
            return None
        else:
            width = int(image.shape[1])
            height = int(image.shape[0])
            # here we consider only the first face found
            x = int(coords[0][0] * width)
            if x < 0:
                x = 0
            y = int(coords[0][1] * height)
            if y < 0:
                y = 0
            w = int(coords[0][2] * width) - x
            h = int(coords[0][3] * height) - y
            # syntax reminder: getRectSubPix(InputArray image, Size patchSize, Point2f center)
            return cv2.getRectSubPix(image, (w, h), (x + w / 2, y + h / 2))

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, outputs):
        coords = []
        for bounding_box in outputs[0][0]:
            conf = bounding_box[2]
            if conf >= THRESHOLD:
                coords.append([bounding_box[3], bounding_box[4], bounding_box[5], bounding_box[6]])
        return coords
