from pathlib import Path
import os
import time
import glob
import sys
from math import ceil
from typing import List
import numpy as np
import pandas as pd
from numpy import ndarray
import pyarrow.feather as ft


class SingleWorker:
    def __init__(self, path: str, common_args: dict):
        self.path = path
        self.common_args = common_args

    def sequential(self):
        self.calculate_and_save_timestamp_differences_fast(self.path, **self.common_args)

    def calculate_differences_2212_fast(self,
                                        data: ndarray,
                                        pixels: List[int],
                                        pix_coor: ndarray,
                                        delta_window: float = 50e3,
                                        cycle_length: float = 4e9,
                                        ):
        """Calculate timestamp differences for firmware version 2212.

        Calculate timestamp differences for the given pixels and LinoSPAD2
        firmware version 2212.

        Parameters
        ----------
        data : ndarray
            Matrix of timestamps, where rows correspond to the TDCs.
        pixels : List[int] | List[List[int]]
            List of pixel numbers for which the timestamp differences should
            be calculated or list of two lists with pixel numbers for peak
            vs. peak calculations.
        pix_coor : ndarray
            Array for transforming the pixel address in terms of TDC (0 to 3)
            to pixel number in terms of half of the sensor (0 to 255).
        delta_window : float, optional
            Width of the time window for counting timestamp differences.
            The default is 50e3 (50 ns).
        cycle_length : float, optional
            Length of each acquisition cycle. The default is 4e9 (4 ms).

        Returns
        -------
        deltas_all : dict
            Dictionary containing timestamp differences for each pair of pixels.

        """

        # Dictionary for the timestamp differences, where keys are the
        # pixel numbers of the requested pairs
        deltas_all = {}

        pixels_left, pixels_right = self.pixel_list_transform(pixels)

        # Find ends of cycles
        cycle_ends = np.argwhere(data[0].T[0] == -2)
        cycle_ends = np.insert(cycle_ends, 0, 0)

        for q in pixels_left:
            # First pixel in the pair
            tdc1, pix_c1 = np.argwhere(pix_coor == q)[0]
            pix1 = np.where(data[tdc1].T[0] == pix_c1)[0]
            for w in pixels_right:
                if w <= q:
                    continue
                deltas_all[f"{q},{w}"] = []

                timestamps_1 = []
                timestamps_2 = []

                # Second pixel in the pair
                tdc2, pix_c2 = np.argwhere(pix_coor == w)[0]
                pix2 = np.where(data[tdc2].T[0] == pix_c2)[0]

                # Go over cycles, shifting the timestamps from each next
                # cycle by lengths of cycles before (e.g., for the 4th cycle
                # add 12 ms)
                for i, _ in enumerate(cycle_ends[:-1]):
                    slice_from = cycle_ends[i]
                    slice_to = cycle_ends[i + 1]
                    pix1_slice = pix1[(pix1 >= slice_from) & (pix1 < slice_to)]
                    if not np.any(pix1_slice):
                        continue
                    pix2_slice = pix2[(pix2 >= slice_from) & (pix2 < slice_to)]
                    if not np.any(pix2_slice):
                        continue

                    # Shift timestamps by cycle length
                    tmsp1 = data[tdc1].T[1][pix1_slice]
                    tmsp1 = tmsp1[tmsp1 > 0]
                    tmsp1 = tmsp1 + cycle_length * i

                    tmsp2 = data[tdc2].T[1][pix2_slice]
                    tmsp2 = tmsp2[tmsp2 > 0]
                    tmsp2 = tmsp2 + cycle_length * i

                    timestamps_1.extend(tmsp1)
                    timestamps_2.extend(tmsp2)

                timestamps_1 = np.array(timestamps_1)
                timestamps_2 = np.array(timestamps_2)

                # Indicators for each pixel: 0 for timestamps from one pixel
                # 1 - from the other
                pix1_ind = np.zeros(len(timestamps_1), dtype=np.int32)
                pix2_ind = np.ones(len(timestamps_2), dtype=np.int32)

                pix1_data = np.vstack((pix1_ind, timestamps_1))
                pix2_data = np.vstack((pix2_ind, timestamps_2))

                # Dataframe for each pixel with pixel indicator and
                # timestamps
                df1 = pd.DataFrame(
                    pix1_data.T, columns=["Pixel_index", "Timestamp"]
                )
                df2 = pd.DataFrame(
                    pix2_data.T, columns=["Pixel_index", "Timestamp"]
                )

                # Combine the two dataframes
                df_combined = pd.concat((df1, df2), ignore_index=True)

                # Sort the timestamps
                df_combined.sort_values("Timestamp", inplace=True)

                # Subtract pixel indicators of neighbors; values of 0
                # correspond to timestamp differences for the same pixel
                # '-1' and '1' - to differences from different pixels
                df_combined["Pixel_index_diff"] = df_combined["Pixel_index"].diff()

                # Calculate timestamp difference between neighbors
                df_combined["Timestamp_diff"] = df_combined["Timestamp"].diff()

                # Get the correct timestamp difference sign
                df_combined["Timestamp_diff"] = (
                        df_combined["Timestamp_diff"] * df_combined["Pixel_index_diff"]
                )

                # Collect timestamp differences where timestamps are from
                # different pixels
                filtered_df = df_combined[
                    abs(df_combined["Pixel_index_diff"]) == 1
                    ]

                # Save only timestamps differences in the requested window
                delta_ts = filtered_df[
                    abs(filtered_df["Timestamp_diff"]) < delta_window
                    ]["Timestamp_diff"].values

                deltas_all[f"{q},{w}"].extend(delta_ts)

        return deltas_all

    def load_calibration_data(self,
                              calibration_path: str,
                              daughterboard_number: str,
                              motherboard_number: str,
                              firmware_version: str,
                              include_offset: bool = False,
                              ):
        """Load the calibration data.

        Parameters
        ----------
        calibration_path : str
            Path to the '.csv' file with the calibration matrix.
        daughterboard_number: str
            The LinoSPAD2 daughterboard number.
        motherboard_number : str
            LinoSPAD2 motherboard (FPGA) number, including the "#".
        firmware_version: str
            LinoSPAD2 firmware version.
        include_offset : bool, optional
            Switch for including the offset calibration. The default is
            True.

        Returns
        -------
        data_matrix : numpy.ndarray
            256x140 matrix containing the calibrated data.
        offset_arr : numpy.ndarray, optional
            Array of 256 offset values, one for each pixel. Returned only if
            include_offset is True.
        """

        path_to_backup = os.getcwd()
        os.chdir(calibration_path)

        # Compensating for TDC nonlinearities
        try:
            file_TDC = glob.glob(f"*TDC_{daughterboard_number}_{motherboard_number}"f"_{firmware_version}*")[0]
        except IndexError as exc:
            raise FileNotFoundError(
                f"TDC calibration for {daughterboard_number}, "
                f"{motherboard_number}, and {firmware_version} is not found") from exc

        # Compensating for offset
        if include_offset:
            try:
                file_offset = glob.glob(f"*Offset_{daughterboard_number}_{motherboard_number}"f"_{firmware_version}*")[
                    0]
            except IndexError:
                raise FileNotFoundError("No .npy file with offset calibration data was found")
            offset_arr = np.load(file_offset)

        # Skipping the first row of TDC bins' numbers
        data_matrix_TDC = np.genfromtxt(file_TDC, delimiter=",", skip_header=1)

        # Cut the first column which is pixel numbers
        data_matrix_TDC = np.delete(data_matrix_TDC, 0, axis=1)

        os.chdir(path_to_backup)

        return (data_matrix_TDC, offset_arr) if include_offset else data_matrix_TDC

    def unpack_binary_data(self,
                           file: str,
                           daughterboard_number: str,
                           motherboard_number: str,
                           firmware_version: str,
                           timestamps: int = 512,
                           include_offset: bool = False,
                           apply_calibration: bool = True,
                           ) -> np.ndarray:
        """Unpacks binary-encoded data from LinoSPAD2 firmware version 2212.

        Parameters
        ----------
        file : str
            Path to the binary data file.
        daughterboard_number : str
            LinoSPAD2 daughterboard number.
        motherboard_number : str
            LinoSPAD2 motherboard (FPGA) number, including the "#".
        firmware_version : str
            LinoSPAD2 firmware version. Either '2212s' (skip) or '2212b' (block).
        timestamps : int, optional
            Number of timestamps per cycle per TDC per acquisition cycle.
            The default is 512.
        include_offset : bool, optional
            Switch for applying offset calibration. The default is True.
        apply_calibration : bool, optional
            Switch for applying TDC and offset calibration. If set to 'True'
            while include_offset is set to 'False', only the TDC
            calibration is applied. The default is True.

        Returns
        -------
        data_all : array-like
            3D array of pixel coordinates in the TDC and the timestamps.

        Raises
        ------
        TypeError
            If 'daughterboard_number', 'motherboard_number', or 'firmware_version'
            parameters are not of string type.
        FileNotFoundError
            If no calibration data file is found.

        Notes
        -----
        The returned data is a 3D array where rows represent TDC numbers,
        columns represent the data, and each cell contains a pixel number in
        the TDC (from 0 to 3) and the timestamp recorded by that pixel.
        """
        # Parameter type check
        if not isinstance(daughterboard_number, str):
            raise TypeError("'daughterboard_number' should be a string.")
        if not isinstance(motherboard_number, str):
            raise TypeError("'motherboard_number' should be a string.")
        if not isinstance(firmware_version, str):
            raise TypeError("'firmware_version' should be a string.")

        # Unpack binary data
        raw_data = np.fromfile(file, dtype=np.uint32)
        # Timestamps are stored in the lower 28 bits
        data_timestamps = (raw_data & 0xFFFFFFF).astype(np.int64)
        # Pixel address in the given TDC is 2 bits above timestamp
        data_pixels = ((raw_data >> 28) & 0x3).astype(np.int8)
        # Check the top bit, assign '-1' to invalid timestamps
        data_timestamps[raw_data < 0x80000000] = -1
        # Free up memory
        del raw_data

        # Number of acquisition cycles in each data file
        cycles = len(data_timestamps) // (timestamps * 65)
        # Transform into a matrix of size 65 by cycles*timestamps
        data_pixels = (
            data_pixels.reshape(cycles, 65, timestamps)
            .transpose((1, 0, 2))
            .reshape(65, -1)
        )

        data_timestamps = (
            data_timestamps.reshape(cycles, 65, timestamps)
            .transpose((1, 0, 2))
            .reshape(65, -1)
        )

        # Cut the 65th TDC that does not hold any actual data from pixels
        data_pixels = data_pixels[:-1]
        data_timestamps = data_timestamps[:-1]

        # Insert '-2' at the end of each cycle
        insert_indices = np.linspace(
            timestamps, cycles * timestamps, cycles
        ).astype(np.int64)

        data_pixels = np.insert(
            data_pixels,
            insert_indices,
            -2,
            1,
        )
        data_timestamps = np.insert(
            data_timestamps,
            insert_indices,
            -2,
            1,
        )

        # Combine both matrices into a single one, where each cell holds pixel
        # coordinates in the TDC and the timestamp
        data_all = np.stack((data_pixels, data_timestamps), axis=2).astype(
            np.int64
        )

        if apply_calibration is False:
            data_all[:, :, 1] = data_all[:, :, 1] * 2500 / 140
        else:
            # Path to the calibration data
            pix_coordinates = np.arange(256).reshape(64, 4)

            path_calibration_data = '/home/sevaedmi/sunrise_testing_script/calibration_data'

            # Include the offset calibration or not
            try:
                if include_offset:
                    calibration_matrix, offset_array = self.load_calibration_data(
                        path_calibration_data,
                        daughterboard_number,
                        motherboard_number,
                        firmware_version,
                        include_offset,
                    )
                else:
                    calibration_matrix = self.load_calibration_data(
                        path_calibration_data,
                        daughterboard_number,
                        motherboard_number,
                        firmware_version,
                        include_offset,
                    )
            except FileNotFoundError:
                raise FileNotFoundError(
                    "No .csv file with the calibration data was found. "
                    "Check the path or run the calibration."
                )

            for i in range(256):
                # Transform pixel number to TDC number and pixel coordinates in
                # that TDC (from 0 to 3)
                tdc, pix = np.argwhere(pix_coordinates == i)[0]
                # Find data from that pixel
                ind = np.where(data_all[tdc].T[0] == pix)[0]
                # Cut non-valid timestamps ('-1's)
                ind = ind[data_all[tdc].T[1][ind] >= 0]
                if not np.any(ind):
                    continue
                data_cut = data_all[tdc].T[1][ind]
                # Apply calibration; offset is added due to how delta ts are
                # calculated
                if include_offset:
                    data_all[tdc].T[1][ind] = (
                            (data_cut - data_cut % 140) * 2500 / 140
                            + calibration_matrix[i, (data_cut % 140)]
                            + offset_array[i]
                    )
                else:
                    data_all[tdc].T[1][ind] = (
                                                      data_cut - data_cut % 140
                                              ) * 2500 / 140 + calibration_matrix[i, (data_cut % 140)]

        return data_all

    def pixel_list_transform(self, pixels: list):
        """Transform a list of pixels into two separate lists.

        Transform the given list of pixels into two separate lists,
        based on the input type (list of integers, of lists, or a mix of
        both).

        Parameters:
            pixels : list
                A list of pixels.

        Returns:
            list: A list of the left pixels.
            list: A list of the right pixels.
        """

        if isinstance(pixels[0], list) and isinstance(pixels[1], list) is True:
            pixels_left, pixels_right = sorted(pixels)
        elif isinstance(pixels[0], int) and isinstance(pixels[1], list) is True:
            pixels_left, pixels_right = sorted([[pixels[0]], pixels[1]])
        elif isinstance(pixels[0], list) and isinstance(pixels[1], int) is True:
            pixels_left, pixels_right = sorted([pixels[0], [pixels[1]]])
        elif isinstance(pixels[0], int) and isinstance(pixels[1], int) is True:
            pixels_left = [pixels[0]]
            pixels_right = [pixels[1]]

        return [pixels_left, pixels_right]


    def file_rewrite_handling(self, file: str, rewrite: bool):
        """Handle file rewriting based on the 'rewrite' parameter.

        The function checks if the specified file exists in the
        'delta_ts_data' directory.
        If it exists and 'rewrite' is True, the file is deleted after a
        countdown (5 sec), this is a good time to stop execution if needed.
        If 'rewrite' is False and the file exists, the function exits with a
        system exit error message.

        Parameters
        ----------
        file : str
            The file path or name to check for existence and potentially
            rewrite.
        rewrite : bool
            If True, the function attempts to rewrite the file after
            deleting it.
            If False and the file already exists, the function exits with an
            error message.

        Returns
        -------
        None

        Raises
        ------
        SystemExit
            If 'rewrite' is False and the file already exists, the function
            exits with an error message.

        """
        try:
            # os.chdir("delta_ts_data")
            if os.path.isfile(file):
                if rewrite is True:
                    print(
                        "\n! ! ! Feather file with timestamps differences already "
                        "exists and will be rewritten ! ! !\n"
                    )
                    for i in range(5):
                        print(f"\n! ! ! Deleting the file in {5 - i} ! ! !\n")
                        time.sleep(1)
                    os.remove(file)
                else:
                    sys.exit(
                        "\n Feather file already exists, 'rewrite' set to"
                        "'False', exiting."
                    )
            # os.chdir("..")
        except FileNotFoundError:
            pass


    def __correct_pix_address(self, pix: int):
        """Pixel address correction.

        Should be used internally only with the "correct_pixel_address"
        function. Transforms the pixel address based on its position.

        Parameters
        ----------
        pix : int
            Pixel address.

        Returns
        -------
        int
            Transformed pixel address.
        """
        if pix > 127:
            pix = 255 - pix
        else:
            pix = pix + 128
        return pix


    def correct_pixels_address(self, pixels: List[int]):
        """Correct pixel address for all given pixels.

        Return the list with the same dimensions as the input.

        Parameters
        ----------
        pixels : List[int] | List[List[int]]
            List of pixel addresses.

        Returns
        -------
        List[int] | List[List[int]]
            List of transformed pixel addresses.
        """
        if isinstance(pixels, list):
            return [self.correct_pixels_address(item) for item in pixels]
        else:
            return self.__correct_pix_address(pixels)


    def apply_mask(self,
                   daughterboard_number: str, motherboard_number: str
                   ) -> np.ndarray:
        """Find and return mask for the requested motherboard.

        Parameters
        ----------
        daughterboard_number : str
            The LinoSPAD2 daughterboard number.
        motherboard_number : str
            LinoSPAD2 motherboard (FPGA) number, including the "#".

        Returns
        -------
        mask : np.ndarray
            The mask array generated from the given daughterboard and motherboard numbers.
        """

        path_to_back = os.getcwd()
        os.chdir('/home/sevaedmi/sunrise_testing_script/masks')
        file_mask = glob.glob(f"*{daughterboard_number}_{motherboard_number}*")[0]
        mask = np.genfromtxt(file_mask).astype(int)
        os.chdir(path_to_back)

        return mask


    def _combine_intermediate_feather_files(self, path: str, skip_data: bool = False):
        """Combine intermediate '.feather' files into one.

        Find all numbered '.feather' files for the data files found in the
        path and combine them all into one.

        Parameters
        ----------
        path : str
            Path to the folder with the '.dat' data files.
        skip_data : bool
            Switch for skipping the data and working directly in the
            'delta_ts_data' folder. Can be used when the raw '.dat' files
            are not available or the data set is incomplete. The default is
            False.

        Raises
        ------
        FileNotFoundError
            Raised when the folder "delta_ts_data", where timestamp
            differences are saved, cannot be found in the path.
        """

        os.chdir(path)

        if not skip_data:
            files_all = sorted(glob.glob("*.dat*"))
            if files_all != []:
                feather_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]
                combined_feather_file_name = feather_file_name
            else:
                feather_file_name = ""
                combined_feather_file_name = "combined"
        else:
            feather_file_name = ""
            combined_feather_file_name = "combined"

        try:
            os.chdir("delta_ts_data")
        except FileNotFoundError:
            raise FileNotFoundError(
                "Folder with saved timestamp differences was not found"
            )

        file_pattern = f"{feather_file_name}*_*.feather"

        feather_files = glob.glob(file_pattern)

        data_combined = []
        data_combined = pd.DataFrame(data_combined)

        for ft_file in feather_files:
            data = ft.read_feather(ft_file)

            data_combined = pd.concat([data_combined, data], ignore_index=True)

            data_combined.to_feather(f"{combined_feather_file_name}.feather")

        for ft_file in feather_files:
            os.remove(ft_file)


    def calculate_and_save_timestamp_differences_fast(self,
            path: str,
            pixels: List[int],
            rewrite: bool,
            daughterboard_number: str,
            motherboard_number: str,
            firmware_version: str,
            timestamps: int = 512,
            delta_window: float = 50e3,
            cycle_length: float = None,
            app_mask: bool = True,
            include_offset: bool = False,
            apply_calibration: bool = True,
            absolute_timestamps: bool = False,
            correct_pix_address: bool = False,
    ):
        """Calculate and save timestamp differences into '.feather' file.

        Unpacks data into a dictionary, calculates timestamp differences for
        the requested pixels, and saves them into a '.feather' table. Works with
        firmware version 2212. Uses a faster algorithm.

        Parameters
        ----------
        path : str
            Path to the folder with '.dat' data files.
        pixels : List[int] | List[List[int]]
            List of pixel numbers for which the timestamp differences should
            be calculated and saved or list of two lists with pixel numbers
            for peak vs. peak calculations.
        rewrite : bool
            switch for rewriting the plot if it already exists. used as a
            safeguard to avoid unwanted overwriting of the previous results.
            Switch for rewriting the '.feather' file if it already exists.
        daughterboard_number : str
            LinoSPAD2 daughterboard number.
        motherboard_number : str
            LinoSPAD2 motherboard (FPGA) number, including the '#'.
        firmware_version: str
            LinoSPAD2 firmware version. Versions "2212s" (skip) and "2212b"
            (block) are recognized.
        timestamps : int, optional
            Number of timestamps per acquisition cycle per pixel. The default
            is 512.
        delta_window : float, optional
            Size of a window to which timestamp differences are compared.
            Differences in that window are saved. The default is 50e3 (50 ns).
        cycle_length: float, optional
            Length of the acquisition cycle. The default is None.
        app_mask : bool, optional
            Switch for applying the mask for hot pixels. The default is True.
        include_offset : bool, optional
            Switch for applying offset calibration. The default is True.
        apply_calibration : bool, optional
            Switch for applying TDC and offset calibration. If set to 'True'
            while apply_offset_calibration is set to 'False', only the TDC
            calibration is applied. The default is True.
        absolute_timestamps: bool, optional
            Indicator for data with absolute timestamps. The default is
            False.
        correct_pix_address : bool, optional
            Correct pixel address for the sensor half on side 23 of the
            daughterboard. The default is False.

        Raises
        ------
        TypeError
            Raised if "pixels" is not a list.
        TypeError
            Raised if "firmware_version" is not a string.
        TypeError
            Raised if "rewrite" is not a boolean.
        TypeError
            Raised if "daughterboard_number" is not a string.
        """
        # Parameter type check
        if isinstance(pixels, list) is False:
            raise TypeError(
                "'pixels' should be a list of integers or a list of two lists"
            )
        if isinstance(firmware_version, str) is False:
            raise TypeError(
                "'firmware_version' should be string, '2212s', '2212b' or '2208'"
            )
        if isinstance(rewrite, bool) is False:
            raise TypeError("'rewrite' should be boolean")
        if isinstance(daughterboard_number, str) is False:
            raise TypeError("'daughterboard_number' should be string")

        os.chdir(path)

        # Handle the input list
        pixels = self.pixel_list_transform(pixels)
        files_all = glob.glob("*.dat")

        files_all = sorted(files_all)

        out_file_name = files_all[0][:-4] + "-" + files_all[-1][:-4]

        # Feather file counter for saving delta ts into separate files
        # of up to 100 MB
        ft_file_number = 0

        # Check if the feather file exists and if it should be rewrited
        feather_file = os.path.join(path, "delta_ts_data", f"{out_file_name}.feather")

        # Remove the old '.feather' files with the pattern
        # for ft_file in feather_files:
        self.file_rewrite_handling(feather_file, rewrite)

        # Go back to the folder with '.dat' files
        os.chdir(path)

        # Define matrix of pixel coordinates, where rows are numbers of TDCs
        # and columns are the pixels that connected to these TDCs
        if firmware_version == "2212s":
            pix_coor = np.arange(256).reshape(4, 64).T
        elif firmware_version == "2212b":
            pix_coor = np.arange(256).reshape(64, 4)
        else:
            print("\nFirmware version is not recognized.")
            sys.exit()

        # Correct pixel addressing for motherboard on side '23'
        if correct_pix_address:
            pixels = self.correct_pixels_address(pixels)

        # Mask the hot/warm pixels
        if app_mask is True:
            mask = self.apply_mask(daughterboard_number, motherboard_number)
            if isinstance(pixels[0], int) and isinstance(pixels[1], int):
                pixels = [pix for pix in pixels if pix not in mask]
            else:
                pixels[0] = [pix for pix in pixels[0] if pix not in mask]
                pixels[1] = [pix for pix in pixels[1] if pix not in mask]

        for i in (range(ceil(len(files_all)))):
            file = files_all[i]

            # Unpack data for the requested pixels into dictionary
            if not absolute_timestamps:
                data_all = self.unpack_binary_data(
                    file,
                    daughterboard_number,
                    motherboard_number,
                    firmware_version,
                    timestamps,
                    include_offset,
                    apply_calibration,
                )
            else:
                data_all, _ = self.unpack_binary_data_with_absolute_timestamps(
                    file,
                    daughterboard_number,
                    motherboard_number,
                    firmware_version,
                    timestamps,
                    include_offset,
                    apply_calibration,
                )

            # If cycle_length is not given manually, estimate from the data
            if cycle_length is None:
                cycle_length = np.max(data_all)

            delta_ts = self.calculate_differences_2212_fast(
                data_all, pixels, pix_coor, delta_window, cycle_length
            )

            # Save data as a .feather file in a cycle so data is not lost
            # in the case of failure close to the end
            delta_ts = pd.DataFrame.from_dict(delta_ts, orient="index")
            delta_ts = delta_ts.T

            try:
                os.chdir("delta_ts_data")
            except FileNotFoundError:
                os.mkdir("delta_ts_data")
                os.chdir("delta_ts_data")

            # Check if feather file exists
            feather_file = f"{out_file_name}_{ft_file_number}.feather"
            if os.path.isfile(feather_file):
                # Check the size of the existing '.feather', if larger
                # than 100 MB, create new one
                if os.path.getsize(feather_file) / 1024 / 1024 < 100:
                    # Load existing feather file
                    existing_data = ft.read_feather(feather_file)

                    # Append new data to the existing feather file
                    combined_data = pd.concat([existing_data, delta_ts], axis=0)
                    ft.write_feather(combined_data, feather_file)
                else:
                    ft_file_number += 1
                    feather_file = f"{out_file_name}_{ft_file_number}.feather"
                    ft.write_feather(delta_ts, feather_file)

            else:
                # Save as a new feather file
                ft.write_feather(delta_ts, feather_file)
            #os.chdir("")

        # Combine the numbered feather files into a single one
        self._combine_intermediate_feather_files(path)

        # Check, if the file was created
        if not (os.path.isfile(path + f"/delta_ts_data/{out_file_name}.feather") is True):
            print("File wasn't generated. Check input parameters.")
