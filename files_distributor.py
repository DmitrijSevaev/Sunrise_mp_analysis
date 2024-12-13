import os
import numpy as np


class FilesDistributor:
    """
    Distribute files in a folder into multiple folders based on the number of chunks

    Parameters
    ----------
    data_path : str
        The path to the folder containing the files to be distributed
    output_base_path : str
        The path to the folder where the distributed files will be stored
    num_of_chunks : int
        The number of chunks to divide the files into
    """

    def __init__(self, data_path, output_base_path, num_of_chunks):
        self.data_path = data_path
        self.output_base_path = output_base_path
        self.num_of_chunks = num_of_chunks

    def distribute(self):
        """
        Distribute the files in the data_path into multiple folders based on the number of chunks
        """
        # find files in the path
        files = os.listdir(self.data_path)

        # Divide files into sub-lists based on the number of chunks
        files = np.array_split(files, self.num_of_chunks)

        # for ech chunk create a new folder and the files
        for i, chunk in enumerate(files):
            folder = os.path.join(self.output_base_path, f'chunk_{i}')
            os.makedirs(folder, exist_ok=True)
            for file in chunk:
                os.rename(os.path.join(self.data_path, file), os.path.join(folder, file))

