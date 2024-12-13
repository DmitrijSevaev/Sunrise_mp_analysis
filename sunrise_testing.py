from files_distributor import FilesDistributor
from files_combiner import FilesCombiner
from single_worker import SingleWorker
import os
import shutil
import datetime
import sys


def clear_dirs(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def restore_from_backup(data_path, backup_path):
    shutil.rmtree(data_path)
    shutil.copytree(backup_path, data_path)


if __name__ == '__main__':
    # Define the paths and parameters
    data_path = 'C:\\Users\\fintv\\Desktop\\CAPADS\\Sunrise\\sunrise_testing_script\\raw_data'
    distributed_dat_path = 'C:\\Users\\fintv\\Desktop\\CAPADS\\Sunrise\\sunrise_testing_script\\distributed_dat_files'
    output_feathers_path = 'C:\\Users\\fintv\\Desktop\\CAPADS\\Sunrise\\sunrise_testing_script\\output_feathers'
    backup_dat_path = 'C:\\Users\\fintv\\Desktop\\CAPADS\\Sunrise\\sunrise_testing_script\\backup_dat_files'
    single_worker_dat_path = distributed_dat_path + '\\chunk_'
    num_of_workers = 10

    # Arguments for delta t calculation
    arguments = {
        "pixels": [144, 171],
        "rewrite": True,
        "daughterboard_number": "NL11",
        "motherboard_number": "#33",
        "firmware_version": "2212b",
        "timestamps": 300,
        "include_offset": False,
    }

    # Clear the directories
    clear_dirs(distributed_dat_path)
    clear_dirs(output_feathers_path)

    start_time = datetime.datetime.now()

    # Distribute the files
    distributor = FilesDistributor(data_path, distributed_dat_path, num_of_workers)
    distributor.distribute()

    restore_from_backup(data_path, backup_dat_path)

    # Process the files
    for i in range(num_of_workers):
        path_do_data = single_worker_dat_path + str(i)
        worker = SingleWorker(path_do_data, arguments)
        worker.calculate_and_save_timestamp_differences_fast(path_do_data, **arguments)
        print(f"Worker {i} done")

    # Combine the files
    combiner = FilesCombiner(distributed_dat_path, output_feathers_path)
    combiner.combine()

    end_time = datetime.datetime.now()

