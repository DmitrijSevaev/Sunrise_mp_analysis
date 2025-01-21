import os
import pandas as pd

class FilesCombiner:
    """
    Combine multiple feather files into a single feather file

    Parameters
    ----------
    intermediate_results : str
        The path to the folder containing the intermediate feather files
    feathers_base_path : str
        The path to the folder where the combined feather file will be saved
    """
    def __init__(self, intermediate_results, feathers_base_path):
        self.intermediate_results = intermediate_results
        self.feathers_base_path = feathers_base_path

    def combine(self):
        """
        Combine multiple feather files into a single feather file
        """
        # Traverse the directory tree to find all .feather files
        feather_files = []
        for root, dirs, files in os.walk(self.intermediate_results):
            for file in files:
                if file.endswith(".feather"):
                    # Construct the full path to the .feather file
                    full_path = os.path.join(root, file)
                    feather_files.append(full_path)

        # Check if any feather files were found
        if not feather_files:
            print("No .feather files found in the directory: " + self.intermediate_results)
            return

        # Use pandas to concatenate all feather files into a single dataframe
        dfs = [pd.read_feather(file) for file in feather_files]
        merged_df = pd.concat(dfs, ignore_index=True)

        # Save the merged dataframe back to a single feather file
        output_file_path = os.path.join(self.feathers_base_path, 'merged.feather')
        merged_df.to_feather(output_file_path)

        # Clear up all the feather files
        for file in feather_files:
            os.remove(file)

