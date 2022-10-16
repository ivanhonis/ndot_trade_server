import datetime
import sys
import os
import time

import threading
from multiprocessing import shared_memory, Lock, Pool, cpu_count
lock = Lock()

import numpy as np
from nDot_trade_server_class.NFolderManager import NFolderManager
from nDot_trade_server_class.NTrader import NTrader


def check_env():
    newpath = os.getcwd() + r"/MESSAGES"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    newpath = os.getcwd() + r"/CONFIGS"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    newpath = os.getcwd() + r"/CONFIGS/nDot_clients.ini"
    if not os.path.exists(newpath):
        print(f"Missing: {newpath}")
        sys.exit()


def worker_folder_manager():
    while True:
        tfm.process_clients()
        tfm.get_status()
        tfm.messages_to_clients()
        time.sleep(25)


def strat_folder_manager():
    forde_manager_thr = threading.Thread(target=worker_folder_manager, args=[])
    forde_manager_thr.start()


if __name__ == "__main__":
    check_env()
    tfm = NFolderManager()
    ntr = NTrader

    # strat_folder_manager()

    # Multiprocess Trade ----------------------------

    shared_array = np.array([1], dtype=np.int64)  # 1 = mehet a trade
    shm = shared_memory.SharedMemory(create=True, size=shared_array.nbytes)
    np_array = np.ndarray(shared_array.shape, dtype=np.int64, buffer=shm.buf)
    np_array[:] = shared_array[:]  # Copy the original data into shared memory
    
    used_cores = cpu_count()
    used_cores = 1
    
    #  Slot manager  :)
    
    slot_fname = f"{tfm.config_dir}/nDot_slots_mp1"
    slots = np.array(["BTCUSDT_P10INT_2D", "ETHUSDT_P10INT_2D"])
    np.save(slot_fname, slots)
    
    params = []
    for x in range(used_cores):
        p_dict = {
            "all_used_cores": used_cores,
            "core": x + 1,
            "shared_memory_name": shm.name,
            "slots_fname": slot_fname + ".npy"
        }
        params.append(p_dict)

    xpool = Pool(used_cores)
    res = xpool.map(ntr, params)



