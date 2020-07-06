from cv2 import imread, imshow, waitKey, destroyAllWindows

image_repo_path = "Datasets\\"


def view_image(image):
    """
    Maintain output window util user presses a key, then all present windows on screen are destroyed.
    :param image::ndarray  -  an array of RGB pixel data.
    """
    imshow('Image', image)
    waitKey(0)
    destroyAllWindows()


def load_image(path):
    # TODO: Add all the image processing function calls.
    image = imread(path)
    view_image(image)

    return True


def start():
    pass


def run_program():
    # TODO: Get the image paths and store them in a list or a dictionary.
    image_paths = ["Datasets\\T6.jpg", "Datasets\\T5.jpg"]
    frame_groups = [load_image(path) for path in image_paths]
    print()


if __name__ == "__main__":
    run_program()
