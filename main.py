import argparse
import logging
import cv2
import time
import os

from face_detection import Face_Detection
from facial_landmarks_detection import Facial_Landmarks_Detection
from gaze_estimation import Gaze_Estimation
from head_pose_estimation import Head_Pose_Estimation
from input_feeder import InputFeeder

# To avoid a very long list of models paths on the command line, here is a list of default models paths.
from mouse_controller import MouseController

FACE_DETECTION_MODEL = "models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001"
GAZE_ESTIMATION_MODEL = "models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002"
HEAD_POSE_ESTIMATION_MODEL = "models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001"
FACIAL_LANDMARKS_DETECTION_MODEL = "models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009"

class Computer_Pointer_Controller:

    def __init__(self, args):

        # load the objects corresponding to the models
        self.face_detection = Face_Detection(args.face_detection_model, args.device, args.extensions, args.perf_counts)
        self.gaze_estimation = Gaze_Estimation(args.gaze_estimation_model, args.device, args.extensions, args.perf_counts)
        self.head_pose_estimation = Head_Pose_Estimation(args.head_pose_estimation_model, args.device, args.extensions, args.perf_counts)
        self.facial_landmarks_detection = Facial_Landmarks_Detection(args.facial_landmarks_detection_model, args.device, args.extensions, args.perf_counts)

        start_models_load_time = time.time()
        self.face_detection.load_model()
        self.gaze_estimation.load_model()
        self.head_pose_estimation.load_model()
        self.facial_landmarks_detection.load_model()
        
        logger = logging.getLogger()
        input_T = args.input_type
        input_F = args.input_file
        
        if input_T.lower() == 'cam':
            # open the video feed
            self.feed = InputFeeder(args.input_type, args.input_file)
            self.feed.load_data()
        else:
            if not os.path.isfile(input_F):
                logger.error('Unable to find specified video file')
                exit(1)
            file_extension = input_F.split(".")[-1]
            if(file_extension in ['jpg', 'jpeg', 'bmp']):
                self.feed = InputFeeder(args.input_type, args.input_file)
                self.feed.load_data()
            elif(file_extension in ['avi', 'mp4']):
                self.feed = InputFeeder(args.input_type, args.input_file)
                self.feed.load_data()
            else:
                logger.error("Unsupported file Extension. Allowed ['jpg', 'jpeg', 'bmp', 'avi', 'mp4']")
                exit(1)
            
        print("Models total loading time :", time.time() - start_models_load_time)

        

        # init mouse controller
        self.mouse_controller = MouseController('low', 'fast')

    def run(self):
        inferences_times = []
        face_detections_times = []
        for batch in self.feed.next_batch():
            if batch is None:
                break

            # as we want the webcam to act as a mirror, flip the frame
            batch = cv2.flip(batch, 1)

            inference_time = time.time()
            face = self.face_detection.predict(batch)
            if face is None:
                logger.error('Unable to detect the face.')
                continue
            else:
                face_detections_times.append(time.time() - inference_time)

                left_eye_image, right_eye_image = self.facial_landmarks_detection.predict(face)
                if left_eye_image is None or right_eye_image is None:
                    continue
                head_pose_angles = self.head_pose_estimation.predict(face)
                if head_pose_angles is None:
                    continue
                vector = self.gaze_estimation.predict(left_eye_image, right_eye_image, head_pose_angles)
                inferences_times.append(time.time() - inference_time)
                if args.show_face == "True":
                    cv2.imshow("Detected face", face)
                    cv2.waitKey(1)
                self.mouse_controller.move(vector[0], vector[1])

        self.feed.close()
        cv2.destroyAllWindows()
        print("Average face detection inference time:", sum(face_detections_times) / len(face_detections_times))
        print("Average total inferences time:", sum(inferences_times) / len(inferences_times))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'The main file to control the mouse pointer from the input. Please execute it with the following arguments.')
    parser.add_argument('--face_detection_model', default=FACE_DETECTION_MODEL, help = 'Path to Face Detection model.')
    parser.add_argument('--gaze_estimation_model', default=GAZE_ESTIMATION_MODEL, help = 'Path to Gaze Estimation model.')
    parser.add_argument('--head_pose_estimation_model', default=HEAD_POSE_ESTIMATION_MODEL , help = 'Path to Head Pose Estimation model.')
    parser.add_argument('--facial_landmarks_detection_model', default=FACIAL_LANDMARKS_DETECTION_MODEL , help = 'Path to Facial Landmark Detection model.')
    parser.add_argument('--device', default='CPU', help = 'The target device to infer on')
    parser.add_argument('--extensions', default=None, help = 'Path to the device extension')
    parser.add_argument('--input_type', default='cam', help = 'Type of input i.e. video file or enter cam for webcam')
    parser.add_argument('--input_file', default=None, help = 'Path to video file, image file or enter cam for webcam')
    parser.add_argument('--show_face', default='True', help = 'Allows to disable the display of the detected face in a window.')
    parser.add_argument('--perf_counts', default='False' , help = 'Displays vital statistics on the terminal about inferences performance for each model used.')

    args = parser.parse_args()

    computer_pointer_controller = Computer_Pointer_Controller(args)
    computer_pointer_controller.run()
