import time
from os import listdir, getcwd
from csv import reader
import xml.etree.ElementTree as xml_tree
import concurrent.futures

_DEFAULT_POOL = concurrent.futures.ThreadPoolExecutor()


# region LOAD SECTION

def file_search(origin_directory_path, paths_map, file_extension):
    """
    Uses the os module to find all the files from the requested type starting at the given directory 
    path - origin_file_path.

    IMPORTANT NOTE: THE SEARCH RELIES ON FINDING A '.' SYMBOL TO DECIDE IF THE ITEM DETECTED IS A FILE OR A DIRECTORY
    TO NOT CONFUSE THE ALGORITHM PLEASE MAKE SURE THERE ARE NO . IN THE NAMES OF THE DIRECTORIES.

    Parameters
    ----------
    (string) origin_directory_path: the path from which the image scan should start.
    (dictionary) images_paths_list: this parameter should be a reference of an empty dictionary to store
    the labels and paths data for each detected image.

    Returns
    -------
    (list) path_list: 3D list containing a 2D list representation of an image, where the innermost list contains
    the B, G, R values of a pixel in an image.
    (list) label_list: the labels of the images in path_list, linked by index.


    """
    try:
        # Lists all the items in the directory with path 'dataset_path'.
        # This could be a list of files, list of directories or list of a combination of both.
        origin_directory = listdir(origin_directory_path)

        # Check if the items in the given origin path are files or directories
        for name in origin_directory:
            new_path = f"{origin_directory_path}\\{name}"

            # If it is a file, store it's path under the file name and continue through the rest of the items.
            if name.endswith(file_extension.lower()) or name.endswith(file_extension.upper()):
                file_name = origin_directory_path.split("\\")[-1]  # Extract the Label from the path.

                # If this file_name exists in the dictionary append the new path
                if file_name in paths_map.keys():
                    paths_map[file_name].append(new_path)
                else:  # Otherwise create the file_name, its list of paths and append.
                    paths_map.update({file_name: [new_path]})

            elif '.' not in name:  # if it is a directory, spawn a recursion to keep searching.
                print("<<file_search::LOG>> Scanning path: ", new_path)
                file_search(new_path, paths_map, file_extension)

    except Exception as e:
        print(f"<<Exception Caught>> {e}")


def load_xml(path, options=""):
    """
    Load the xml file into a tree-like structure.

    Parameters
    ----------
    (string) path: path to the file containing the data.
    (string) options: Currently not used. Would have to pass options to the xml file as well because it is in the same
                       dictionary as the csv load. Can use this options variable later to customise how to parse the file.
    Returns
    -------
    (Element Object) root: The root node of the XML Tree.
    """
    return xml_tree.parse(path).getroot()


# Returns a file object
def load_txt(path, options="r"):
    return open(path, options)


# This function will return the reader object
def load_csv(path, options="r"):
    return reader(open(path, options))


# region EXTRA TASKS FOR FUNCTION
# TODO: Add more file formats: XAML, JSON, etc.
#
# TODO: Add configuration options to the function.
# 		1) Choosing between different formats for the final dataset structure:
# 			1.1) Grouping by rows, columns, attributes.
# 		2) Specifying which part of the file to read.
# 			2.1) Where to start and where to end.
# 			2.2) From a specific character index to another character index.
#
# TODO: Look at using the * operator to pass any number of options as one construct.
# endregion
def load_file(file_type, file_path, options="r"):
    # TODO: assert that file_type is a string

    # Implemented exception handling for loading files.
    try:
        loaded_files = {"XML": load_xml, "TXT": load_txt, "CSV": load_csv}
        return loaded_files[file_type.upper()](file_path, options)

    except IOError:  # The exception thrown is IOError, check if my try catch will stop it
        print("\n" + 117 * "-" + "\n" \
                                 "\tNO SUCH FILE OR DIRECTORY: Current Working Directory  - ", getcwd(), " \n" \
                                 "\tPlease make sure your data set is stored within the 'data' directory " \
                                 "in your current working directory!" \
                                 "\n" + 117 * "-" + "\n")

        return None

# endregion


#region DESIGN PATTERNS SECTION
class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.
    """

    def __init__(self, decorated):
        self._decorated = decorated

    def get_instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance

        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise Exception('the class is decorated as Singleton, Singletons must be accessed through `get_instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)
#endregion


# region Performance Tracking and Improvements Section
# Metrics to profile: Speed(Time), Calls(Frequency)
def time_function(fun):
    def wrap(*pos_args, **kw_args):
        start = time.time()
        returned = fun(*pos_args, **kw_args)
        print("[", fun.__name__, " function] Time taken: ", round(time.time() - start, 2), "seconds")
        return returned

    return wrap


def threadpool(f, executor=None):
    def wrap(*args, **kwargs):
        return (executor or _DEFAULT_POOL).submit(f, *args, **kwargs)

    return wrap

# endregion


# region Others
def screen_to_viewport_space(position, image_boundaries):
    """
    Gets the viewport space percentage of a range 0 to image_boundaries for a given point in that range.

    Parameters
    ----------
    (list/tuple) position: Example (x), (x,y), (x,y,z)
    (list/tuple) image_boundaries: Example (1080), (1080, 720), (1080, 720, 200)

    Return
    ------
    (list) percentages: a list of the equivalent percentage of the given max boundaries.
           Example: When given point with (x=540, y=180) on max boundaries (x=1080, y=720),
           the output percentages will be (x=0.5, y=0.25).
    """
    percentages = []
    for i in range(len(position)):
        percentages.append(position[i] / image_boundaries[i])

    return percentages


def viewport_to_screen_space(percentage, image_boundaries):
    """
    Gets the screen space position of a range 0 to image_boundaries for a given point in that range.

    Parameters
    ----------
    (list/tuple) percentage: Example (x%), (x%,y%), (x%,y%,z%)
    (list/tuple) image_boundaries: Example (1080), (1080, 720), (1080, 720, 200)

    Return
    ------
    (list) positions: a list of the equivalent percentage of the given max boundaries.
           Example: When given point with (x=0.5, y=0.25) on max boundaries (x=1080, y=720),
           the output positions will be (x=540, y=180).
    """
    positions = []
    for i in range(len(percentage)):
        positions.append(int(round(percentage[i] * image_boundaries[i])))

    return tuple(positions)

# endregion
