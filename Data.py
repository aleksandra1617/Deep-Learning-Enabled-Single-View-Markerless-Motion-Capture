from cv2 import *


class Data:
    _instance = None
    default_model = None

    @staticmethod
    def get_instance():
        if Data._instance is None:
            Data()
        return Data._instance

    def __init__(self):
        """ Virtually private constructor. """
        if Data._instance is not None:
            raise Exception("Instancing not permitted, the class KeypointDetector is a Singleton! " +
                            "Please call the get_instance function instead!")
        else:
            Data._instance = self

    #region CONFIG SECTION
    def load_video(self):
        pass

    def extract_frames(self):
        pass

    def configure(self, default_model):
        """
        Sets the configuration of the class at start so that fewer parameter passes are required on function calls.
        :param default_model:
        :return:
        """
        pass
    #endregion
