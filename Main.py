from Utilities import file_search, screen_to_viewport_space, viewport_to_screen_space
from JointDetection import JointDetector
import cv2
import numpy as np


DEFAULT_DATA_REPO_PATH = "Datasets\\"
POINT_DETECTOR_CNN = None
VIDEO_RESOLUTION = None


# region DATA LOADING
def load_video_frames(video_file_path):
    global VIDEO_RESOLUTION
    """
    Extracts the frames from a video file through OpenCV and retains the file name because the dictionary is
    unordered meaning the key is the only way to find which skeleton/output file belongs to which video/input file.
    The name extraction is done by utilising the built in string functions and indexing on the first line of the
    function.

    Parameters
    ----------
    (string) video_file_path: the path to a training input file.

    Returns
    -------
    (list) file_name:
    (list) video_frames:
    """
    file_name = video_file_path.split("\\")[-1].split(".")[0][:-4]
    capture = cv2.VideoCapture(video_file_path)
    frame_count = capture.get(cv2.CAP_PROP_FPS)

    video_frames = []
    for count in range(int(frame_count)):
        captured, frame = capture.read()
        # Check if the current frame has a different resolution from the one set.
        if (VIDEO_RESOLUTION == frame.shape) or (VIDEO_RESOLUTION is None):
            VIDEO_RESOLUTION = frame.shape

            if captured:
                # TODO: assert that the captured frame is not null/none
                video_frames.append(frame)

        else:
            print("Frame ", count, " in Video << ", file_name, " >> has resolution << ", frame.shape,
                  " >> which does not match the VIDEO_RESOLUTION set at the start of data loading. "
                  "Please resize the frames to << ", VIDEO_RESOLUTION, " >>!")

    #video_frames = np.asarray(video_frames)
    #video_frames = np.reshape(video_frames, (30, 1080, 1920, 3))


    video_frames = np.asarray(video_frames)
    video_frames = video_frames.astype('float32')/255.0

    return file_name, video_frames


def load_skeleton_data(skeleton_file_path):
    """
    https://medium.com/@lisajamhoury/understanding-kinect-v2-joints-and-coordinate-system-4f4b90b9df16

    Parameters
    ----------
    (string) skeleton_file_path: the path to a training output file.
    """
    # Extracting the file name from the path string
    file_name, point_list = skeleton_file_path.split("\\")[-1].split(".")[0], []

    file_reader = open(skeleton_file_path, 'r')
    skeleton_file_data = list(file_reader)[4:]
    file_reader.close()

    # This loop extracts the data from 1 skeleton each iteration.
    start_index, end_index, offset, num_key_points = 0, 25, 3, 25
    joint_order = []

    while end_index < len(skeleton_file_data):
        current_skeleton = skeleton_file_data[start_index:end_index]
        skeleton = []
        for i in range(len(current_skeleton)):
            joint = current_skeleton[i]   # a joint is represented by a line in the skeleton file

            array = joint.rstrip().split(" ")
            if len(array) < 6:
                print("Sh*t!!!")
                break

            else:

                x = float(joint.rstrip().split(" ")[5])
                y = float(joint.rstrip().split(" ")[6])

                if (x < 800) and (y < 300):
                    print("Bad input key points!")

                assert(y <= VIDEO_RESOLUTION[0] and x <= VIDEO_RESOLUTION[1]), "<<Warning>> Illegal joint position, " \
                                                                               "key point out of frame."

                key_point = screen_to_viewport_space([round(x), round(y)], (VIDEO_RESOLUTION[1], VIDEO_RESOLUTION[0]))
                skeleton += key_point

        #skeleton = skeleton[:30]
        start_index = end_index+offset
        end_index = start_index+num_key_points
        point_list.append(skeleton)
        skeleton = []

    #np.reshape(point_list, (74, 25, 2))
    point_list = point_list[:30]
    point_list = np.asarray(point_list)
    point_list = point_list.astype('float32')
    return file_name, point_list


def load_dataset(input_file_extension, output_file_extension, dataset_name="", ):
    print("\n<<Main::LOG>> Loading dataset.. Default data path is '", DEFAULT_DATA_REPO_PATH + dataset_name, "'")

    # Find all input and output files TODO: Let the user modify the path without the need to go into this.
    input_data_paths, output_data_paths = {}, {}
    print("\n<<Main::LOG>> Search for files with extension .avi commencing..")
    file_search(DEFAULT_DATA_REPO_PATH + dataset_name, input_data_paths, input_file_extension)
    print("\n<<Main::LOG>> Search for files with extension .skeleton commencing..")
    file_search(DEFAULT_DATA_REPO_PATH + dataset_name, output_data_paths, output_file_extension)
    print("\n<<Main::LOG>> File Path Scan Complete.")

    print("\n<<Main::LOG>> Starting Frame Extraction..")
    video_frames, key_points = {}, {}

    # Goes through each dataset key containing input data paths.
    for path_list in input_data_paths.values():
        for path in path_list:
            file_name, frame_list = load_video_frames(path)
            video_frames.update({file_name: frame_list})

    for path_list in output_data_paths.values():
        for path in path_list:
            file_name, point_list = load_skeleton_data(path)
            key_points.update({file_name: point_list})

    return video_frames, key_points


def render_video_with_joints(input_videos, output_joints):
    for i in range(len(input_videos)):
        frame = input_videos[i]
        frame_key_points = output_joints

        # Convert the percentage based locations to screen space positions.
        """for j in range(len(output_joints[i])):
            percentage = output_joints[i][j]
            key_point = viewport_to_screen_space(percentage, (VIDEO_RESOLUTION[1], VIDEO_RESOLUTION[0]))
            output_joints[i][j] = key_point"""

        # Link the key points with lines TODO: build a json that will let me map instantly!
        frame = cv2.line(frame, frame_key_points[0], frame_key_points[1], (206, 232, 0), 5)  # sBase->sMid
        frame = cv2.line(frame, frame_key_points[2], frame_key_points[3], (206, 232, 0), 5)  # neck->head

        frame = cv2.line(frame, frame_key_points[4], frame_key_points[5], (206, 232, 0), 5)  # lShoulder->lElbow
        frame = cv2.line(frame, frame_key_points[5], frame_key_points[6], (206, 232, 0), 5)  # lElbow->lWrist
        frame = cv2.line(frame, frame_key_points[6], frame_key_points[7], (206, 232, 0), 5)  # lWrist->lHand

        frame = cv2.line(frame, frame_key_points[8], frame_key_points[9], (206, 232, 0), 5)  # rShoulder->rElbow
        frame = cv2.line(frame, frame_key_points[9], frame_key_points[10], (206, 232, 0), 5)  # rElbow->rWrist
        frame = cv2.line(frame, frame_key_points[10], frame_key_points[11], (206, 232, 0), 5)  # rWrist->rHand

        frame = cv2.line(frame, frame_key_points[12], frame_key_points[13], (206, 232, 0), 5)  # lHip->lKnee
        frame = cv2.line(frame, frame_key_points[13], frame_key_points[14], (206, 232, 0), 5)  # lKnee->lAnkle
        frame = cv2.line(frame, frame_key_points[14], frame_key_points[15], (206, 232, 0), 5)  # lAnkle->lFoot

        frame = cv2.line(frame, frame_key_points[16], frame_key_points[17], (206, 232, 0), 5)  # rHip->rKnee
        frame = cv2.line(frame, frame_key_points[17], frame_key_points[18], (206, 232, 0), 5)  # rKnee->rAnkle
        frame = cv2.line(frame, frame_key_points[18], frame_key_points[19], (206, 232, 0), 5)  # rAnkle->rFoot

        frame = cv2.line(frame, frame_key_points[20], frame_key_points[4], (206, 232, 0), 5) # sShoulder->lShoulder
        frame = cv2.line(frame, frame_key_points[20], frame_key_points[8], (206, 232, 0), 5)  # sShoulder->rShoulder
        frame = cv2.line(frame, frame_key_points[20], frame_key_points[2], (206, 232, 0), 5) # sShoulder->neck
        frame = cv2.line(frame, frame_key_points[20], frame_key_points[1], (206, 232, 0), 5)  # sShoulder->neck

        frame = cv2.line(frame, frame_key_points[0], frame_key_points[12], (206, 232, 0), 5)  # sBase->rHip
        frame = cv2.line(frame, frame_key_points[0], frame_key_points[16], (206, 232, 0), 5)  # sBase->rShoulder

        frame = cv2.line(frame, frame_key_points[7], frame_key_points[22], (206, 232, 0), 5)  # lHand->lThumb
        frame = cv2.line(frame, frame_key_points[7], frame_key_points[21], (206, 232, 0), 5)  # lHand->lhTip

        frame = cv2.line(frame, frame_key_points[11], frame_key_points[24], (206, 232, 0), 5)  # rHand->rThumb
        frame = cv2.line(frame, frame_key_points[11], frame_key_points[23], (206, 232, 0), 5)  # rHand->rhTip

        # Map all the key points on top of the lines on the image
        for j in range(len(frame_key_points)-1):
            current_key_point = frame_key_points[j]
            frame = cv2.circle(frame, current_key_point, 6, (0, 127, 255), -1)

        if i == 0 or i == 14 or i == 18 or i == 29:
            cv2.imshow('Video'+', Frame '+str(i), frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def start_program():
    """
    On start program the load_dataset function is called to allow the user to select a path for the dataset.
    :return:
    """
    input_videos, output_joints = load_dataset(".avi", ".skeleton", "NTURGBD\\Sample")
    print("<<LOG>> Data Load Complete!")

    # Display a video with its skeleton data to check if it is all loaded correctly.
    # render_video_with_joints(input_videos, output_joints)

    # Set the configuration for the learning model and trains it on the loaded data.
    start_model_generation(input_videos, output_joints)

    """
    for file_name in input_videos:
        for i in range(len(input_videos[file_name])):
            frame = input_videos[file_name][i]
            frame_key_points = output_joints[file_name][i]
            resolution = frame.shape"""


def start_model_generation(loaded_in_data, loaded_out_data):
    """
    Instantiates a singleton for the JointDetector class which handles constructing a deep learning model
    for human 2D rig key point identification.

    Parameters
    ----------
    (dictionary) training_data:
    """
    global POINT_DETECTOR_CNN

    # Instantiating the required classes
    # Now that the data is loaded it is time to instantiate the classes that will use it.
    POINT_DETECTOR_CNN = JointDetector.get_instance()

    # Load config files and configure the instances.
    POINT_DETECTOR_CNN.configure(joint_data=50, frame_shape=VIDEO_RESOLUTION, batch_size=2, num_epochs=12)

    # Split the data into train and test.
    slice_index = round(len(loaded_in_data) * 0.7)

    input_training = list(loaded_in_data.values())[:slice_index]
    input_testing = list(loaded_in_data.values())[slice_index:]
    output_training = list(loaded_out_data.values())[:slice_index]
    output_testing = list(loaded_out_data.values())[slice_index:]

    train, test = {"input": input_training, "output": output_training}, {"input": input_testing, "output": output_testing}

    # Generate the model
    POINT_DETECTOR_CNN.create_model(train, test)
    print('<<LogTest>> Traning output: ', train['output'][0][0])
    output = JointDetector.get_instance().predict(train["input"][0])
    final_output = []
    print('#' * 50)
    for frame in output[0]:
        for i in range(0, len(frame), 2):
            final_output.append(viewport_to_screen_space((frame[i], frame[i+1]), (1920, 1080)))

    render_video_with_joints(test["input"][0], final_output)


def extract_motion():
    # TODO: Resize the input video for prediction to the set Video Resolution.
    # POINT_DETECTOR_CNN.predict(videos)
    # TODO: Playback the video.
    pass


def main():
    start_program()


if __name__ == "__main__":
    main()
