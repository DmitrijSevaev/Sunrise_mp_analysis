import os
import pandas as pd


class FilesCombiner:
    """
    Combine multiple feather files into a single feather file

    Parameters
    ----------
    feathers_base_path : str
        The path to the folder containing the feather files to be combined
    """
    def __init__(self, intermediate_results, feathers_base_path):
        self.intermediate_results = intermediate_results
        self.feathers_base_path = feathers_base_path

    def combine(self):
        """
        Combine multiple feather files into a single feather file
        """
        # Traverse the directory tree
        feather_files = []
        for root, dirs, files in os.walk(self.intermediate_results):
            for file in files:
                if file.endswith(".feather"):
                    # Construct the full path to the .feather file
                    full_path = os.path.join(root, file)
                    feather_files.append(full_path)

        # Use pandas to concatenate all feather files into a single dataframe
        dfs = [pd.read_feather(file) for file in feather_files]
        merged_df = pd.concat(dfs, ignore_index=True)

        # Save the merged dataframe back to a single feather file
        output_file_path = self.feathers_base_path + '/merged.feather'
        merged_df.to_feather(output_file_path)

        # clear up all the feather files
        for file in feather_files:
            os.remove(file)
