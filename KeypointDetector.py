# Confidence and affinity maps are outputted from the first layer.
import numpy as np
import cv2
import glob


class KeypointDetector:
    _instance = None
    default_model = None

    @staticmethod
    def get_instance():
        if KeypointDetector._instance is None:
            KeypointDetector()
        return KeypointDetector._instance

    def __init__(self):
        """ Virtually private constructor. """
        if KeypointDetector._instance is not None:
            raise Exception("Instancing not permitted, the class KeypointDetector is a Singleton! " +
                            "Please call the get_instance function instead!")
        else:
            KeypointDetector._instance = self

    # region CONFIG SECTION
    def configure(self, default_model):
        """
        Sets the configuration of the class at start so that fewer parameter passes are required on function calls.
        :param default_model:
        :return:
        """
        KeypointDetector.default_model = default_model
    #endregion


def load_trained_model():
    """
    The default train model is ...
    :return: 
    """
    pass


def retrain_with(dataset):
    """
    :param dataset: the dataset to train on,
    :return:
    """
    pass


if __name__ == "__main__":
    kd = KeypointDetector.get_instance()
    k = KeypointDetector.get_instance()
    print(kd)
    print(k)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
