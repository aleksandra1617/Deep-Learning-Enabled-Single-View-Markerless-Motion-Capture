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
    file_name, point_list, original_point_list = skeleton_file_path.split("\\")[-1].split(".")[0], [], []

    file_reader = open(skeleton_file_path, 'r')
    skeleton_file_data = list(file_reader)[4:]
    file_reader.close()

    # This loop extracts the data from 1 skeleton each iteration.
    start_index, end_index, offset, num_key_points = 0, 25, 3, 25
    joint_order = []

    while end_index < len(skeleton_file_data):
        # A chunk of the skeleton file that contains 25 lines and each one represents the data for a single joint.
        current_skeleton_read = skeleton_file_data[start_index:end_index]

        skeleton_viewport, skeleton_screen_space = [], []
        for i in range(len(current_skeleton_read)):
            joint = current_skeleton_read[i]   # A joint is represented by a line in the skeleton file

            joint_array = joint.rstrip().split(" ")
            if len(joint_array) < 6:
                print("The joint data array extracted is too small, check that the data is passed in correctly!")
                break

            else:
                x = float(joint.rstrip().split(" ")[5])
                y = float(joint.rstrip().split(" ")[6])

                assert(x <= VIDEO_RESOLUTION[1] and y <= VIDEO_RESOLUTION[0]), "<<Warning>> Illegal joint position, " \
                                                                               "key point out of frame."
                skeleton_screen_space += [x, y]

                key_point = screen_to_viewport_space([round(x), round(y)], (VIDEO_RESOLUTION[1], VIDEO_RESOLUTION[0]))
                skeleton_viewport += key_point

        # Validating output positions
        for i in range(0, len(skeleton_screen_space), 2):
            expected_width = skeleton_screen_space[i] / skeleton_viewport[i]
            expected_height = skeleton_screen_space[i+1] / skeleton_viewport[i+1]

            if (expected_width > 1922) or (expected_height > 1082):  # Allowed margin error 2.0
                print("LOAD_SKELETON<<WARNING>> Inaccurate conversion from screen space to view port position!")

        start_index = end_index + offset
        end_index = start_index + num_key_points
        point_list.append(skeleton_viewport)
        original_point_list.append(skeleton_screen_space)
        skeleton_viewport = []

    point_list = point_list[:30]
    point_list = np.asarray(point_list)
    point_list = point_list.astype('float32')

    return file_name, point_list, original_point_list[:30]


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
    video_frames, key_points, original_key_points = {}, {}, {}

    # Goes through each dataset key containing input data paths.
    for path_list in input_data_paths.values():
        for path in path_list:
            file_name, frame_list = load_video_frames(path)
            video_frames.update({file_name: frame_list})

    for path_list in output_data_paths.values():
        for path in path_list:
            file_name, point_list, original_point_list = load_skeleton_data(path)
            key_points.update({file_name: point_list})
            original_key_points.update({file_name: original_point_list})

    return video_frames, key_points, original_key_points


def render_video_with_joints(input_data, output_data):
    for video_count in range(len(input_data)):
        video = input_data[video_count]
        skeletons = output_data[video_count]

        for frame_count in range(len(video)):
            frame = video[frame_count]
            skeleton_joints = skeletons[frame_count]

            # Format the skeleton data so that it can be given to OpenCV for render.
            formatted_skeleton_joints = []
            for point_count in range(0, len(skeleton_joints), 2):
                # Convert the percentage based locations to screen space positions.
                key_point = viewport_to_screen_space((skeleton_joints[point_count],
                                                      skeleton_joints[point_count + 1]),
                                                      (1920, 1080))

                formatted_skeleton_joints.append(key_point)
                #formatted_skeleton_joints.append((skeleton_joints[point_count], skeleton_joints[point_count + 1]))

            if frame_count == 0 or frame_count == 29:  # frame_count == 14 or frame_count == 18 or
                # Link the key points with lines TODO: build a json that will let me map instantly!
                frame = cv2.line(frame, formatted_skeleton_joints[0], formatted_skeleton_joints[1], (232, 206, 0), 5)  # sBase->sMid
                frame = cv2.line(frame, formatted_skeleton_joints[2], formatted_skeleton_joints[3], (232, 206, 0), 5)  # neck->head

                frame = cv2.line(frame, formatted_skeleton_joints[4], formatted_skeleton_joints[5], (232, 206, 0), 5)  # lShoulder->lElbow
                frame = cv2.line(frame, formatted_skeleton_joints[5], formatted_skeleton_joints[6], (232, 206, 0), 5)  # lElbow->lWrist
                frame = cv2.line(frame, formatted_skeleton_joints[6], formatted_skeleton_joints[7], (232, 206, 0), 5)  # lWrist->lHand

                frame = cv2.line(frame, formatted_skeleton_joints[8], formatted_skeleton_joints[9], (232, 206, 0), 5)  # rShoulder->rElbow
                frame = cv2.line(frame, formatted_skeleton_joints[9], formatted_skeleton_joints[10], (232, 206, 0), 5)  # rElbow->rWrist
                frame = cv2.line(frame, formatted_skeleton_joints[10], formatted_skeleton_joints[11], (232, 206, 0), 5)  # rWrist->rHand

                frame = cv2.line(frame, formatted_skeleton_joints[12], formatted_skeleton_joints[13], (232, 206, 0), 5)  # lHip->lKnee
                frame = cv2.line(frame, formatted_skeleton_joints[13], formatted_skeleton_joints[14], (232, 206, 0), 5)  # lKnee->lAnkle
                frame = cv2.line(frame, formatted_skeleton_joints[14], formatted_skeleton_joints[15], (232, 206, 0), 5)  # lAnkle->lFoot

                frame = cv2.line(frame, formatted_skeleton_joints[16], formatted_skeleton_joints[17], (232, 206, 0), 5)  # rHip->rKnee
                frame = cv2.line(frame, formatted_skeleton_joints[17], formatted_skeleton_joints[18], (232, 206, 0), 5)  # rKnee->rAnkle
                frame = cv2.line(frame, formatted_skeleton_joints[18], formatted_skeleton_joints[19], (232, 206, 0), 5)  # rAnkle->rFoot

                frame = cv2.line(frame, formatted_skeleton_joints[20], formatted_skeleton_joints[4], (232, 206, 0), 5) # sShoulder->lShoulder
                frame = cv2.line(frame, formatted_skeleton_joints[20], formatted_skeleton_joints[8], (232, 206, 0), 5)  # sShoulder->rShoulder
                frame = cv2.line(frame, formatted_skeleton_joints[20], formatted_skeleton_joints[2], (232, 206, 0), 5) # sShoulder->neck
                frame = cv2.line(frame, formatted_skeleton_joints[20], formatted_skeleton_joints[1], (232, 206, 0), 5)  # sShoulder->neck

                frame = cv2.line(frame, formatted_skeleton_joints[0], formatted_skeleton_joints[12], (232, 206, 0), 5)  # sBase->rHip
                frame = cv2.line(frame, formatted_skeleton_joints[0], formatted_skeleton_joints[16], (232, 206, 0), 5)  # sBase->rShoulder

                frame = cv2.line(frame, formatted_skeleton_joints[7], formatted_skeleton_joints[22], (232, 206, 0), 5)  # lHand->lThumb
                frame = cv2.line(frame, formatted_skeleton_joints[7], formatted_skeleton_joints[21], (232, 206, 0), 5)  # lHand->lhTip

                frame = cv2.line(frame, formatted_skeleton_joints[11], formatted_skeleton_joints[24], (232, 206, 0), 5)  # rHand->rThumb
                frame = cv2.line(frame, formatted_skeleton_joints[11], formatted_skeleton_joints[23], (232, 206, 0), 5)  # rHand->rhTip

                # Map all the key points on top of the lines on the image
                for j in range(len(formatted_skeleton_joints)-1):
                    current_key_point = formatted_skeleton_joints[j]
                    frame = cv2.circle(frame, current_key_point, 6, (0, 127, 255), -1)

                cv2.imshow('Video'+str(video_count)+', Frame '+str(frame_count), frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def start_program():
    """
    On start program the load_dataset function is called to allow the user to select a path for the dataset.
    :return:
    """
    input_videos, output_joints, original_output_joints = load_dataset(".avi", ".skeleton", "NTURGBD\\Sample")
    print("<<LOG>> Data Load Complete!")

    # Display a video with its skeleton data to check if it is all loaded correctly.
    render_video_with_joints(list(input_videos.values()), list(output_joints.values()))

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

    render_video_with_joints(test["input"][0], output)


def extract_motion():
    # TODO: Resize the input video for prediction to the set Video Resolution.
    # POINT_DETECTOR_CNN.predict(videos)
    # TODO: Playback the video.
    pass


def main():
    start_program()


if __name__ == "__main__":
    main()
