import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.io import loadmat
import random
import math
from numpy import genfromtxt
import resampy
import matplotlib.pyplot as plt
import pdb
#have to put the time to predict value here.
TIMETOPREDICT= 1.5

# this is not where the data is, but where we will save the processed data
data_root = Path("processed_data_updated/version_1/")
#these separation were used when we were doing transfer learning and divided the real data in parts.
data_milton_par = Path("new_realdata/cut_TL_Train")
 
data_milton_pred1 = Path("new_realdata/cut_TL_Test") 
data_milton_pred2 = Path("new_realdata/moreTest/test4") 

def read_tsv(tsv, start_index = 1024, end_index = None, scale = True):
    """
    Process tab-separated values (TSV) data from a file.

    Parameters:
        tsv (str): The path to the TSV file to read.
        start_index (int, optional): The starting index for data extraction. Default is 1024.
        end_index (int, optional): The ending index for data extraction. Default is None, which extracts data to the end.
        scale (bool, optional): If True, scale the extracted data; otherwise, return it as is. Default is True.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - phi (numpy.ndarray): Processed angular data.
            - x1 (numpy.ndarray): Processed x1 data.
    """
    # Read the TSV data and split it into six columns
    dat = np.genfromtxt(tsv, delimiter='\t')
    x1, x2, x3, x4, x5, x6 = np.hsplit(dat, 6)
    
    # Calculate ell_1, angle, and apply reshaping
    ell_1 = np.sqrt((x1 - x4) ** 2 + (x3 - x6) ** 2) + 1e-10
    ang_sin_x_1 = (x1 - x4) / ell_1
    phi = np.arcsin(ang_sin_x_1) * 180 / np.pi
    phi = phi.reshape(-1)[start_index:end_index]
    x1 = x4.reshape(-1)[start_index:end_index]

    # Resample the data from 250 Hz to 100 Hz
    phi = resampy.resample(phi, sr_orig=250, sr_new=100)
    x1 = resampy.resample(x1, sr_orig=250, sr_new=100)

    # Scale the data if 'scale' is True
    if scale:
        phi = phi / 20
        x1 = x1 / 0.335

    return phi, x1

def strides(a, window_len=128, stride=1):
    """
    Generate overlapping windows from a 1D array or list.

    Parameters:
        a (array-like): The input array or list.
        window_len (int, optional): The length of the window. Default is 128.
        stride (int, optional): The step size between windows. Default is 1.

    Returns:
        array-like: A 2D array with overlapping windows.

    If the input 'a' is a list, it will be converted to a NumPy array for processing.
    If 'stride' is set to None, it will be automatically determined based on the array size.
    """
    to_list = False

    if isinstance(a, list):
        to_list = True
        a = np.array(a)

    if stride is None:
        stride = math.ceil(a.size / 10)

    nrows = ((a.size - window_len) // stride) + 1
    n = a.strides[0]
    windows = np.lib.stride_tricks.as_strided(a, shape=(nrows, window_len), strides=(stride * n, n))

    if to_list:
        windows = windows.tolist()

    return windows

def getAllMats(path="new_data/folder001", sample=None, filter_required=True, L=128, timetopredict=TIMETOPREDICT, samplingrate=100):
    """
    Retrieve a list of .mat files from the specified path.

    Parameters:
        path (str, optional): The directory path to search for .mat files. Default is "new_data/folder001".
        sample (int, optional): Number of files to sample from the list. Default is None (no sampling).
        filter_required (bool, optional): If True, filter files based on criteria. Default is True.
        L (int, optional): Length parameter. Default is 128.
        timetopredict (int): The time to predict in seconds.
        samplingrate (int): The sampling rate in Hz.

    Returns:
        list: A list of .mat file paths.

    If 'filter_required' is True, files are filtered based on certain criteria before returning the list.
    """
    # Calculate the maximum allowed data size based on 'timetopredict' and 'samplingrate'
    maxsize = int(timetopredict * samplingrate) + L * 4 + 1

    # Find all .mat files in the specified directory
    data = sorted(Path(path).glob("milton*.mat")

    # If 'sample' is specified, randomly sample 'sample' files from the list
    if sample:
        data = random.sample(data, sample)

    if filter_required:
        # Apply filters to the data based on certain criteria
        data = list(filter(lambda x: loadmat(x)['phiv'].max() > 20, data))

    return data


def prepareData(path, cutoff=75, delay=23, L=128, timetopredict=TIMETOPREDICT, samplingrate=100, fall_stride=1, start_index=1024, end_index=None, scale=True):
    """
    Prepare and preprocess data for machine learning.

    Args:
        path (str or Path): Path to the original data file (.mat or .tsv).
        cutoff (int, optional): Number of data points to cut off from the end. Default is 75.
        delay (int, optional): Delay parameter. Default is 23.
        L (int, optional): Window size. Default is 128.
        timetopredict (int): Time to predict in seconds.
        samplingrate (int): The sampling rate in Hz.
        fall_stride (int, optional): Step size for fall data. Default is 1.
        start_index (int, optional): Starting index for data extraction. Default is 1024.
        end_index (int, optional): Ending index for data extraction.

    Returns:
        tuple: A tuple containing the preprocessed data:
            - x (numpy.ndarray): Input features.
            - y (numpy.ndarray): Output labels.
            - S (int): Stride size.

    The function can load and preprocess data from both .mat and .tsv files.
    It applies various transformations to create input features and output labels for machine learning.

    Note: This code assumes the existence of some external variables like 'TIMETOPREDICT.'
    """
    # Define a function to split an array into two parts
    def split_array(arr, separator):
        return arr[:-separator], arr[-separator:]

    if path.suffix == ".mat":
        if not isinstance(path, str):
            path = path.as_posix()
        mat = loadmat(path)
        phiv = mat['phiv'].reshape(-1)
        dxv = mat['xv'].reshape(-1)
        if scale:
            phiv = phiv / 20
            dxv = dxv / 0.335
    elif path.suffix == ".tsv":
        phiv, dxv = read_tsv(path, start_index=start_index, end_index=end_index)

    # Trim data by cutting off 'cutoff' points from the end
    phiv = phiv[:-cutoff]
    dxv = dxv[:-cutoff]

    # Compute delayed and response data for phiv and dxv
    delay_phiv = phiv[delay:]
    respo_phiv = phiv[:-delay]
    delay_dxv = dxv[delay:]
    respo_dxv = dxv[:-delay]

    # Calculate the size of the fall region
    fall_region = L + int(timetopredict * samplingrate)
    
    # Split delayed and response data into non-fall and fall regions
    non_fall_phiv, fall_phiv = split_array(delay_phiv, fall_region)
    non_fall_dxv, fall_dxv = split_array(delay_dxv, fall_region)

    # Determine the stride size 'S' based on the data
    if non_fall_phiv.size < fall_phiv.size:
        S = 1
    else:
        fall_window = (fall_phiv.size - L) / 1 + 1
        S = math.ceil((non_fall_phiv.size - L) / (fall_window - 1))
        
    # Perform striding on non-fall and fall data
    non_fall_phiv = strides(non_fall_phiv, L=L, S=S)
    non_fall_dxv = strides(non_fall_dxv, L=L, S=S)
    fall_phiv = strides(fall_phiv, L=L, S=fall_stride)
    fall_dxv = strides(fall_dxv, L=L, S=fall_stride)
    
    # Randomly downsample the non-fall data
    np.random.seed(42)
    random_indices = np.random.choice(non_fall_phiv.shape[0], size=10, replace=False)
    non_fall_phiv = non_fall_phiv[random_indices]
    non_fall_dxv = non_fall_dxv[random_indices]

    # Create feature and label arrays
    x = np.concatenate((non_fall_phiv, non_fall_dxv, fall_phiv, fall_dxv), axis=-1)
    y = np.concatenate((np.zeros(non_fall_phiv.shape[0]), np.ones(fall_phiv.shape[0]))

    return shuffle(x, y), S

def getOneHugeArray(path_list, start_index_list = None, end_index_list = None):
    """
    Combine data from multiple sources into a single large array.

    Args:
        path_list (list): List of data file paths to process.
        start_index_list (list, optional): List of starting indices for data extraction. Default is None.
        end_index_list (list, optional): List of ending indices for data extraction. Default is None.

    Returns:
        tuple: A tuple containing the combined data:
            - x (numpy.ndarray): Input features.
            - y (numpy.ndarray): Output labels.

    This function processes data from a list of files and combines them into a single large array. It uses the 'prepareData' function to prepare and preprocess each data source and concatenate the results.

    Note: If 'start_index_list' and 'end_index_list' are not provided, the entire data is processed. If these lists are provided, data is extracted based on the indices specified.
    """
    datax = []
    datay = []
    if start_index_list is None and end_index_list is None:    
        for path in tqdm(path_list):
            x,y,_  = prepareData(path, cutoff = 75, delay = 23, L = 128, timetopredict = TIMETOPREDICT, samplingrate = 100, fall_stride = 1)
            datax.append(x)
            datay.append(y)
    else:
        for path, start_index, end_index in tqdm(zip(path_list, start_index_list, end_index_list)):
            x,y,_  = prepareData(path, cutoff = 75, delay = 23, L = 128, timetopredict =TIMETOPREDICT, samplingrate = 100, fall_stride = 1, start_index = start_index, end_index = end_index)
            datax.append(x)
            datay.append(y)
    x = np.concatenate(datax)    
    y = np.concatenate(datay)

    return x, y

def create_Real_Data(data_path_list, mode="pred"):
    """
    Create real data plots, combine data, and save as numpy archive.

    Args:
        data_path_list (list): List of data file paths.
        mode (str, optional): The mode for data creation. Default is "pred."

    This function generates plots for the data in 'data_path_list,' combines the data into a single array, and saves it as a NumPy archive.

    Note: The 'data_path_list' should contain paths to TSV files.
    """
    # Sort and filter the data path list
    data_path_list = sorted(data_path_list.glob('*.tsv'))
    start_index_list = [1024] * len(data_path_list)
    end_index_list = [None] * len(data_path_list)

    # Plot the data
    for path in tqdm(data_path_list):
        phi, _ = read_tsv(path)
        plt.plot(phi, label=path.name)

    plt.legend()
    plt.savefig(f"plots_{mode}.png")

    # Combine and save the data
    x, y = getOneHugeArray(data_path_list, start_index_list=start_index_list, end_index_list=end_index_list)
    print("Data Shape:", x.shape, y.shape)
    print("Class Distribution:", y.sum() / y.size)

    np.savez(data_root / f"{mode}.npz", x=x, y=y)

if __name__ == '__main__':
    # Create and save real data with different modes
    create_Real_Data(data_milton_pred1, mode="pred1")
    create_Real_Data(data_milton_pred2, mode="pred2")
    create_Real_Data(data_milton_par, mode="par")

    # Get a list of simulated data paths
    sim_path_list = getAllMats(sample=None)
    print("The sampled sim_path_list is", len(sim_path_list))

    # Train-test split for simulated data
    train_path_list = sim_path_list[:-int(len(sim_path_list) * 0.05)]
    val_path_list = sim_path_list[-int(len(sim_path_list) * 0.05):]
    testsim_path_list = sim_path_list[-int(len(val_path_list) * 0.05):]

    # Load and preprocess data for training, validation, and testing
    xtrain, ytrain = getOneHugeArray(train_path_list)
    print(f"The shape of xtrain is {xtrain.shape}, the ratio of fall/nofall: {ytrain.sum() / ytrain.size}")
    
    xval, yval = getOneHugeArray(val_path_list)
    print(f"The shape of xval is {xval.shape}, the ratio of fall/nofall: {yval.sum() / yval.size}")

    xtest, ytest = getOneHugeArray(testsim_path_list)
    print(f"The shape of xtest is {xtest.shape}, the ratio of fall/nofall: {ytest.sum() / ytest.size}")

    # Create a directory for data storage if it doesn't exist
    data_root.mkdir(exist_ok=True)

    # Save training, validation, and testing data
    np.savez(data_root / "train.npz", x=xtrain, y=ytrain)
    np.savez(data_root / "val.npz", x=xval, y=yval)
    np.savez(data_root / "test.npz", x=xtest, y=ytest)

    # Create a metadata file
    meta = """
    ONE-BIG-DATA for all
    Data is sampled at 100,
    Train test split is done by taking first 90 for train and 10 for test
    We use dynamic stride size in the fall region, no fall region uses only stride = 1
    Window size = 128
    Data being used is from simulated data: "newchaos/data/mats/synthetic",
    these data are filtered to remove size less than 1024 and that doesn't have a fall (angle should be greater than 20 degrees).
    """
    with open(data_root / "metadata.txt", "w") as f:
        f.write(meta)