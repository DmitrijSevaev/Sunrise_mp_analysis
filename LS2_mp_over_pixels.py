""" #TODO

"""

import glob
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from daplis.functions import calc_diff as cd
from daplis.functions import utils
from daplis.functions.calibrate import load_calibration_data
from pyarrow import feather as ft


class MpWizard:

    # Initialize by passing the input parameters which later will be
    # passed into all internal functions
    def __init__(
        self,
        path: str = "",
        pixels: list = [],
        daughterboard_number: str = "",
        motherboard_number: str = "",
        firmware_version: str = "",
        timestamps: int = 512,
        delta_window: float = 50e3,
        include_offset: bool = False,
        apply_calibration: bool = True,
        apply_mask: bool = True,
        absolute_timestamps: bool = False,
        number_of_cores: int = 1,
    ):

        self.path = path
        self.pixels = pixels
        self.daughterboard_number = daughterboard_number
        self.motherboard_number = motherboard_number
        self.firmware_version = firmware_version
        self.timestamps = timestamps
        self.delta_window = delta_window
        self.include_offset = include_offset
        self.apply_calibration = apply_calibration
        self.apply_mask = apply_mask
        self.absolute_timestamps = absolute_timestamps
        self.number_of_cores = number_of_cores

        os.chdir(self.path)

        # Load calibration if requested
        if self.apply_calibration:

            work_dir = Path(__file__).resolve().parent.parent

            # TODO
            # path_calibration_data = os.path.join(
            #     work_dir, r"params\calibration_data"
            # )
            path_calibration_data = r"C:\Users\bruce\Documents\GitHub\daplis\src\daplis\params\calibration_data"
            # path_calibration_data = (
            #     r"/home/sj/GitHub/daplis/src/daplis/params/calibration_data"
            # )

            calibration_data = load_calibration_data(
                path_calibration_data,
                daughterboard_number,
                motherboard_number,
                firmware_version,
                include_offset,
            )

            if self.include_offset:
                self.calibration_matrix, self.offset_array = calibration_data
            else:
                self.calibration_matrix = calibration_data

        # Apply mask if requested
        if self.apply_mask:
            mask = utils.apply_mask(
                self.daughterboard_number,
                self.motherboard_number,
            )
            if isinstance(self.pixels[0], int) and isinstance(
                self.pixels[1], int
            ):
                self.pixels = [pix for pix in self.pixels if pix not in mask]
            else:
                self.pixels = [
                    [value for value in sublist if value not in mask]
                    for sublist in pixels
                ]

        # Check the firmware version and set the pixel coordinates accordingly
        if self.firmware_version == "2212s":
            self.pix_coor = np.arange(256).reshape(4, 64).T
        elif firmware_version == "2212b":
            self.pix_coor = np.arange(256).reshape(64, 4)
        else:
            print("\nFirmware version is not recognized.")
            sys.exit()

    def _unpack_binary_data(
        self,
        file: str,
    ) -> np.ndarray:
        """Unpack binary data from LinoSPAD2.

        Return a 3D matrix of pixel numbers and timestamps.

        Parameters
        ----------
        file : str
            A '.dat' data file from LinoSPAD2 with the binary-encoded
            data.

        Returns
        -------
        np.ndarray
            A 3D matrix of the pixel numbers where timestamp was
            recorded and timestamps themselves.
        """
        # Unpack binary data
        raw_data = np.memmap(file, dtype=np.uint32)
        # Timestamps are stored in the lower 28 bits
        data_timestamps = (raw_data & 0xFFFFFFF).astype(np.int64)
        # Pixel address in the given TDC is 2 bits above timestamp
        data_pixels = ((raw_data >> 28) & 0x3).astype(np.int8)
        # Check the top bit, assign '-1' to invalid timestamps
        data_timestamps[raw_data < 0x80000000] = -1

        # Number of acquisition cycles in each data file
        cycles = len(data_timestamps) // (self.timestamps * 65)
        # Transform into a matrix of size 65 by cycles*timestamps
        data_pixels = (
            data_pixels.reshape(cycles, 65, self.timestamps)
            .transpose((1, 0, 2))
            .reshape(65, -1)
        )

        data_timestamps = (
            data_timestamps.reshape(cycles, 65, self.timestamps)
            .transpose((1, 0, 2))
            .reshape(65, -1)
        )

        # Cut the 65th TDC that does not hold any actual data from pixels
        data_pixels = data_pixels[:-1]
        data_timestamps = data_timestamps[:-1]

        # Insert '-2' at the end of each cycle
        insert_indices = np.linspace(
            self.timestamps, cycles * self.timestamps, cycles
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

        if self.apply_calibration is False:
            data_all[:, :, 1] = data_all[:, :, 1] * 2500 / 140
        else:
            # Path to the calibration data
            pix_coordinates = np.arange(256).reshape(64, 4)
            for i in range(256):
                # Transform pixel number to TDC number and pixel
                # coordinates in that TDC (from 0 to 3)
                tdc, pix = np.argwhere(pix_coordinates == i)[0]
                # Find data from that pixel
                ind = np.where(data_all[tdc].T[0] == pix)[0]
                # Cut non-valid timestamps ('-1's)
                ind = ind[data_all[tdc].T[1][ind] >= 0]
                if not np.any(ind):
                    continue
                data_cut = data_all[tdc].T[1][ind]
                # Apply calibration; offset is added due to how delta
                # ts are calculated
                if self.include_offset:
                    data_all[tdc].T[1][ind] = (
                        (data_cut - data_cut % 140) * 2500 / 140
                        + self.calibration_matrix[i, (data_cut % 140)]
                        + self.offset_array[i]
                    )
                else:
                    data_all[tdc].T[1][ind] = (
                        data_cut - data_cut % 140
                    ) * 2500 / 140 + self.calibration_matrix[
                        i, (data_cut % 140)
                    ]

        return data_all

    def _calculate_timestamps_differences(self, args):
        """Calculate photon coincidences and save to '.feather'.

        Parameters
        ----------
        #TODO
        """

        # try-except railguard for a function that goes to separate
        # cores
        file, data, pixel_pair = args
        try:
            # Check if the 'delta_ts_data' folder exists
            output_dir = Path(self.path) / "delta_ts_data"
            output_dir.mkdir(exist_ok=True)

            # Calculate the differences and convert them to a pandas
            # dataframe
            deltas_all = cd.calculate_differences_2212_fast(
                data, pixel_pair, self.pix_coor
            )
            data_for_plot_df = pd.DataFrame.from_dict(
                deltas_all, orient="index"
            ).T

            file_name = Path(file).stem
            output_file = (
                output_dir
                / f"{file_name}_{pixel_pair[0]}_{pixel_pair[1]}.feather"
            )
            ft.write_feather(
                data_for_plot_df.reset_index(drop=True), output_file
            )

        except Exception as e:
            print(f"Error processing file {file}: {e}")

        result = {f"{pixel_pair[0]},{pixel_pair[1]}": deltas_all}

        return result

    def _combine_feather_files(self, path_to_feather_files: str):

        os.chdir(path_to_feather_files)

        for pixel_pair in self.pixel_pairs:
            ft_files = glob.glob(f"*{pixel_pair[0]}_{pixel_pair[1]}*.feather")
            data_all = pd.DataFrame()
            for ft_file in ft_files:
                data = ft.read_feather(ft_file)
                data_all = pd.concat((data_all, data), ignore_index=True)
            data_all.to_feather(
                f"combined_{pixel_pair[0]}_{pixel_pair[1]}.feather"
            )

    def calculate_and_save_timestamp_differences_mp(self):

        # Find all LinoSPAD2 data files
        files = glob.glob("*.dat")

        if not files:
            raise ValueError("No .dat files found in the specified path.")

        self.pixel_pairs = []
        for i in self.pixels[0]:
            for j in self.pixels[1]:
                self.pixel_pairs.append([i, j])

        print(self.pixel_pairs)

        start_time = time.time()

        for file in files:
            print(f"Processing file: {file}")
            data = self._unpack_binary_data(file)

            # Via Pool

            # args = [
            #     (file, data, pixel_pair) for pixel_pair in self.pixel_pairs
            # ]

            # with multiprocessing.Pool(min(self.number_of_cores, len(self.pixel_pairs))) as pool:
            #     pool.map(self._calculate_timestamps_differences, args)

            # Via Process
            processes = []

            # Create one process per pixel pair for the current file
            for pixel_pair in self.pixel_pairs:
                args = (file, data, pixel_pair)
                p = multiprocessing.Process(
                    target=self._calculate_timestamps_differences, args=(args,)
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

        end_time = time.time()

        print(
            f"Parallel processing of files "
            "files (with each writing to its file) finished "
            f"in: {round(end_time - start_time, 2)} s"
        )

        # Combine '.feather' files from separate cores
        path_to_feathers = os.path.join(self.path, "delta_ts_data")

        self._combine_feather_files(path_to_feathers)

        path_output = os.path.join(self.path, "delta_ts_data")

        print(
            "The feather files with the timestamp differences were "
            f"combined into the 'combined.feather' file in {path_output}"
        )


if __name__ == "__main__":
    path = r"D:\LinoSPAD2\Data\B7d\Ne640\Ne640\5_file_test"

    mpboii = MpWizard(
        path,
        pixels=[[142, 143, 144], [170, 172, 173]],
        daughterboard_number="B7d",
        motherboard_number="#28",
        firmware_version="2212s",
        timestamps=700,
        number_of_cores=20,
    )

    mpboii.calculate_and_save_timestamp_differences_mp()


### Sequential

import time

from daplis.functions import delta_t

time_start = time.time()

path = r"D:\LinoSPAD2\Data\B7d\Ne640\Ne640\5_file_test"

delta_t.calculate_and_save_timestamp_differences_fast(
    path,
    rewrite=True,
    pixels=[[142, 143, 144], [170, 172, 173]],
    daughterboard_number="B7d",
    motherboard_number="#28",
    firmware_version="2212s",
    timestamps=700,
)

print(f"Finished in {time.time() - time_start}")
