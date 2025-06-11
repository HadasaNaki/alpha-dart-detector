"""
Utility module for loading DICOM and RTSS files for SPECT imaging analysis.
"""

import os
import pydicom
import tkinter as tk
from tkinter import filedialog
from typing import Tuple, List, Optional, Union, Any

from SPECT_Data import SPECTData
from RTSS import RTSS


def load_dicom_and_rtss_this_project(
    path: str = "",
    needed_organs: Union[List[str], str] = "all",
    improve_resolution_factor: Tuple[int, int, int] = (1, 1, 1)
) -> Tuple[SPECTData, Any]:
    """
    Load both DICOM and RTSS files from a directory.

    Parameters:
        path (str): Directory containing SPECT DICOM and RTSS files
        needed_organs (Union[List[str], str]): Specific organs to load or "all"
        improve_resolution_factor (Tuple[int, int, int]): Resolution improvement factor

    Returns:
        Tuple[SPECTData, Any]: Tuple containing SPECT data and VOI data
    """
    spect_data = load_dicom(path)
    voi_data = load_rtss(path, spect_data, needed_organs, improve_resolution_factor)
    return spect_data, voi_data


def load_dicom(path: str = "") -> Optional[SPECTData]:
    """
    Load SPECT DICOM files from a given directory or through file dialog.

    Parameters:
        path (str): Directory containing SPECT DICOM files or path to a specific file

    Returns:
        Optional[SPECTData]: An instance of SPECTData or None if loading fails
    """
    # If no path provided, prompt user with file dialog
    if not path:
        path = select_file_dialog(
            title="Select DICOM File",
            filetypes=[("DICOM files", "*.dcm")]
        )
        if not path:
            return None  # User canceled file selection

    # Check if path is a directory or a file
    if os.path.isdir(path):
        # Process as directory - search for SPECT files
        return _load_dicom_from_directory(path)
    else:
        # Process as direct file
        try:
            ds = pydicom.dcmread(path)
            return SPECTData(dicom_dataset=ds)
        except Exception as e:
            print(f"Error loading DICOM file: {e}")
            return None


def _load_dicom_from_directory(path: str) -> Optional[SPECTData]:
    """
    Search through directories to find SPECT DICOM files.

    Parameters:
        path (str): Root directory to search for SPECT files

    Returns:
        Optional[SPECTData]: SPECT data if found, None otherwise
    """
    try:
        directories = os.listdir(path)

        # Find the SPECT images in subdirectories
        for directory in directories:
            dir_path = os.path.join(path, directory)
            if not os.path.isdir(dir_path):
                continue

            files = os.listdir(dir_path)
            if not files:
                continue

            # Load first file to check if it's a SPECT image
            sample_file = os.path.join(dir_path, files[0])
            try:
                dicom_dataset = pydicom.dcmread(sample_file, force=True)

                # Check if this is a SPECT image
                if (hasattr(dicom_dataset, 'Modality') and
                    dicom_dataset.Modality == 'NM' and
                    hasattr(dicom_dataset, 'ImageID') and
                    'TOMO' in dicom_dataset.ImageID):
                    return SPECTData(dicom_dataset=dicom_dataset)
            except Exception as e:
                print(f"Error reading file {sample_file}: {e}")
                continue

        print("No SPECT DICOM files found in the specified directory")
        return None
    except Exception as e:
        print(f"Error searching directory: {e}")
        return None


def load_rtss(
    path: str = "",
    spect_data: Optional[SPECTData] = None,
    needed_organs: Union[List[str], str] = "all",
    improve_resolution_factor: Tuple[int, int, int] = (1, 1, 1),
    debug: bool = True
) -> Optional[Any]:
    """
    Load RTSS (RT Structure Set) files either from a directory or a specific file.

    Parameters:
        path (str): Directory or file path for RTSS
        spect_data (Optional[SPECTData]): SPECT data object for correlation
        needed_organs (Union[List[str], str]): List of organs to load or "all"
        improve_resolution_factor (Tuple[int, int, int]): Resolution improvement factor
        debug (bool): Enable debug output

    Returns:
        Optional[Any]: VOI data or None if loading fails
    """
    # Validate spect_data
    if spect_data is None:
        print("Error: SPECT data must be provided to load RTSS")
        return None

    # Check if path is a specific file
    if os.path.isfile(path) and path.lower().endswith('.dcm'):
        return RTSS(
            path,
            spect_data,
            needed_organs=needed_organs,
            debug=debug,
            improve_resolution_factor=improve_resolution_factor
        )

    # If path is a directory, search for RTSS files
    if os.path.isdir(path):
        return _load_rtss_from_directory(
            path,
            spect_data,
            needed_organs,
            improve_resolution_factor,
            debug
        )

    # If no path provided, prompt user with file dialog
    if not path:
        rtss_path = select_file_dialog(
            "Select RTSS File",
            [("DICOM RTSS files", "*.dcm")]
        )
        if not rtss_path:
            return None  # User canceled file selection

        return RTSS(
            rtss_path,
            spect_data,
            needed_organs=needed_organs,
            debug=debug,
            improve_resolution_factor=improve_resolution_factor
        )

    return None


def _load_rtss_from_directory(
    path: str,
    spect_data: SPECTData,
    needed_organs: Union[List[str], str],
    improve_resolution_factor: Tuple[int, int, int],
    debug: bool
) -> Optional[Any]:
    """
    Search through directory to find RTSS files.

    Parameters:
        path (str): Root directory to search for RTSS files
        spect_data (SPECTData): SPECT data object
        needed_organs (Union[List[str], str]): List of organs to load or "all"
        improve_resolution_factor (Tuple[int, int, int]): Resolution improvement factor
        debug (bool): Enable debug output

    Returns:
        Optional[Any]: VOI data if found, None otherwise
    """
    try:
        directories = os.listdir(path)

        # Look for ContourNM directory which typically contains RTSS files
        for directory in directories:
            dir_path = os.path.join(path, directory)
            if not os.path.isdir(dir_path):
                continue

            if 'ContourNM' in dir_path:
                files = os.listdir(dir_path)
                if not files:
                    continue

                # Use the first RTSS file found
                rtss_path = os.path.join(dir_path, files[0])
                return RTSS(
                    rtss_path,
                    spect_data,
                    needed_organs=needed_organs,
                    debug=debug,
                    improve_resolution_factor=improve_resolution_factor
                )

        print("No RTSS files found in the specified directory")
        return None
    except Exception as e:
        print(f"Error searching for RTSS: {e}")
        return None


def select_file_dialog(title: str, filetypes: List[Tuple[str, str]]) -> str:
    """
    Display a file selection dialog.

    Parameters:
        title (str): Dialog window title
        filetypes (List[Tuple[str, str]]): File types to display

    Returns:
        str: Selected file path or empty string if canceled
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path