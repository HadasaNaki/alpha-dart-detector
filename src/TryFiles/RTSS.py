import numpy as np
import random
import datetime
from pathlib import Path
import pydicom as dicom
from pydicom.dataset import Dataset
import cv2 as cv
from skimage.draw import polygon
from SPECT_Data import SPECTData
import copy
import os



class RTSS:
    """
    Class for handling and manipulating RT Structure Set (RTSS) DICOM files.

    RT files contain detailed information about Regions of Interest (ROIs) that are overlaid onto DICOM images.
    The RT structure is primarily composed of two key sequences:
    - StructureSetROISequence: Registers each ROI (name, number, etc.)
    - ROIContourSequence: Contains the boundary data for each ROI.

    For each slice and every connected component within the slice, a list of physical coordinates (x, y, z)
    is provided to define the contour points which represent the precise boundaries of the ROI in 3D space.

    Attributes:
        rt_file_path (str): Path to the RT DICOM file.
        rt_ds (Dataset): DICOM dataset loaded from the RT file.
        output_dir (str): Directory to save output data like contour images.
        image_shape (tuple): Shape of the image with improved resolution.
        first_pixel_position (np.ndarray): Physical position of the first pixel in the image.
        axis_spacing (np.ndarray): Pixel spacing in the x, y, z directions.
        transform_matrix (np.ndarray): Matrix for transforming between physical and index coordinates.
        debug (bool): Flag for enabling debug output.
        voi_info (list): Information about each ROI including name, number, contour points, and mask.
        voi_array (np.ndarray): 3D array representing all volumes of interest.
        colors_used (list): List of colors used for displaying ROIs.
    """

    def __init__(self, rt_file_path, spect_data, needed_organs="all",
                 improve_resolution_factor=(1, 1, 1), output_dir="contour_test",
                 handling_overlap=False, debug=False):
        """
        Initialize the RTSS object and extract contour data from the RT DICOM file.

        In study cases, the RT is already adjusted for the SPECT physical domain.

        Parameters:
            rt_file_path (str): Path to the RT DICOM file.
            spect_data (SPECTData): Object containing image shape, position, pixel spacing, and orientation.
            needed_organs (str or list): Specific organs to include, "all" for all organs.
            improve_resolution_factor (tuple): Factor to improve resolution of the SPECT image.
                Default (1,1,1) maintains original resolution.
                Set to (8,8,1) for more accurate results at the expense of processing time.
            output_dir (str): Directory to save output images. Default is "contour_test".
            handling_overlap (bool): Whether to handle overlapping regions between ROIs.
            debug (bool): Flag to enable debugging output.
        """
        # Initialize basic attributes
        self.colors_used = [[0, 0, 0], [255, 255, 255]]  # Background and foreground colors
        self.rt_file_path = rt_file_path
        self.rt_ds = dicom.dcmread(rt_file_path)
        self.output_dir = output_dir
        self.debug = debug

        # Set up image dimensions and spacing
        self.improve_resolution_factor = np.array(improve_resolution_factor)
        self.image_shape = tuple(np.array(spect_data.image_shape) * np.roll(self.improve_resolution_factor, shift=1))
        self.first_pixel_position = np.array(spect_data.image_position)
        self.axis_spacing = spect_data.pixel_spacing / self.improve_resolution_factor

        # Handle negative z-direction if needed
        if self.first_pixel_position[2] > 0:
            # If slices are arranged from head to top, z decreases
            self.axis_spacing[2] = -self.axis_spacing[2]

        # Set up transformation matrix
        self._transform_matrix(spect_data.image_orientation)

        # Extract and process contour data
        self.voi_info = self._extract_contour_data()
        self.voi_array = np.zeros(self.image_shape, dtype=np.uint8)
        self._reconstruct_all_voi_mask(needed_organs, handling_overlap)

    def copy_instance(self):
        """
        Create a deep copy of the current RTSS instance.

        Returns:
            RTSS: A deep copy of the current RTSS instance.
        """
        return copy.deepcopy(self)

    def _extract_contour_data(self):
        """
        Extract contour data and relevant ROI information from the RT DICOM file.

        Returns:
            list: List of dictionaries, each containing:
                - 'num': ROI number
                - 'name': ROI name
                - 'color': Display color of the ROI
                - 'contour_points': List of contour points for each slice
        """
        all_voi_info = []

        for roi_sequence in self.rt_ds.StructureSetROISequence:
            roi_number = roi_sequence.ROINumber
            roi_name = roi_sequence.ROIName

            # Find matching contour data in ROIContourSequence
            roi_contour_sequence = self.rt_ds.ROIContourSequence[roi_number - 1]
            roi_color = getattr(roi_contour_sequence, 'ROIDisplayColor', [0, 255, 0])  # Default color is green
            self.colors_used.append(roi_color)

            # Extract contour point data
            contour_data = [item.ContourData for item in roi_contour_sequence.ContourSequence]

            # Create ROI info dictionary
            roi_info = {
                'num': roi_number,
                'name': roi_name,
                'color': roi_color,
                'contour_points': contour_data  # List of contours
            }
            all_voi_info.append(roi_info)

        return all_voi_info

    def _transform_matrix(self, image_orientation):
        """
        Generate the transformation matrix from the image orientation data.

        In most cases, this results in an identity matrix and can be neglected.

        Parameters:
            image_orientation (np.ndarray): A 1x6 array defining the image orientation in DICOM standard.
        """
        image_orientation = np.reshape(image_orientation, (2, 3))
        row_cosines = image_orientation[0]
        col_cosines = image_orientation[1]
        slice_normal = np.cross(row_cosines, col_cosines)
        self.transform_matrix = np.vstack([row_cosines, col_cosines, slice_normal]).T

    def _transform_physical_to_indexes(self, contour):
        """
        Convert physical coordinates of contours to index coordinates in the image space.

        Parameters:
            contour (np.ndarray): List of physical coordinates [x,y,z].

        Returns:
            np.ndarray: Array of index coordinates, rounded to integer values.
        """
        # Convert to matrix of points, each row is another point
        contour_physical = np.reshape(contour, newshape=(-1, 3))

        # Translate coordinates relative to first pixel position
        contour_physical -= self.first_pixel_position

        # When the transform_matrix is I (identity), simplified calculation
        if np.allclose(self.transform_matrix, np.eye(3)):
            contour_indexes = contour_physical / self.axis_spacing
        else:
            # Full transformation when transform_matrix is not identity
            contour_indexes = np.dot(self.transform_matrix, contour_physical.T).T / self.axis_spacing

        # Indexes must be integers
        contour_indexes = np.round(contour_indexes).astype(int)
        return contour_indexes

    def _transform_indexes_to_physical(self, contour_indexes):
        """
        Convert index coordinates back to physical coordinates in the original image space.

        Parameters:
            contour_indexes (np.ndarray): Array of index coordinates.

        Returns:
            np.ndarray: Array of physical coordinates.
        """
        if contour_indexes.size == 0:
            return np.array([])

        # Scale index coordinates by axis spacing
        contour_physical = contour_indexes.astype(float) * self.axis_spacing

        # Apply transformation if transform_matrix is not identity
        if not np.allclose(self.transform_matrix, np.eye(3)):
            contour_physical = np.dot(self.transform_matrix, contour_indexes.T).T

        # Translate back to absolute coordinates
        contour_physical += self.first_pixel_position

        return contour_physical.round(3)

    def voi_mask(self, voi_name):
        """
        Retrieve the 3D mask for a specific Volume of Interest (VOI) by name.

        Parameters:
            voi_name (str): Name of the VOI to retrieve.

        Returns:
            np.ndarray: Boolean 3D mask of the VOI if found.

        Raises:
            AttributeError: If the specified VOI name is not found.
        """
        search_list = [voi for voi in self.voi_info if voi['name'] == voi_name]

        if len(search_list) == 1:
            voi_data = search_list[0]
            return self._reconstruct_voi_mask(voi_data)
        else:
            raise AttributeError(f"The VOI '{voi_name}' was not found")

    def _reconstruct_voi_mask(self, voi_data):
        """
        Reconstruct the VOI mask from contour data for a specific VOI.

        Parameters:
            voi_data (dict): Dictionary of specific VOI containing its contour points.

        Returns:
            np.ndarray: VOI mask as a boolean numpy array.
        """
        voi_mask = np.zeros_like(self.voi_array, dtype=bool)
        contours = voi_data['contour_points']

        for contour in contours:
            # Convert physical coordinates to index coordinates
            contour_indexes = self._transform_physical_to_indexes(contour)

            # All points of a contour are in the same slice
            slice_idx = contour_indexes[0, 2]

            # Define the ROI region using polygon filling
            rr, cc = polygon(contour_indexes[:, 1], contour_indexes[:, 0], shape=voi_mask.shape[1:])
            voi_mask[slice_idx, rr, cc] = True

        return voi_mask

    def _is_needed_organ(self, voi_data, needed_organs):
        """
        Check if the VOI is in the list of needed organs.

        Parameters:
            voi_data (dict): VOI data dictionary.
            needed_organs (list or str): List of needed organ names or "all".

        Returns:
            bool: True if the VOI is needed, False otherwise.
        """
        if needed_organs == "all":
            return True

        for organ in needed_organs:
            if organ.lower() in voi_data["name"].lower():
                # Skip tumors without numeric identifiers when checking for "tumor"
                if "tumor" in voi_data["name"].lower() and not any(char.isdigit() for char in voi_data["name"]):
                    continue
                return True

        return False

    def _reconstruct_all_voi_mask(self, needed_organs, handling_overlap=False):
        """
        Reconstruct 3D masks for all VOIs and handle overlap between them if required.

        Parameters:
            needed_organs (list or str): List of organs to include or "all" for all organs.
            handling_overlap (bool): Flag to enable or disable overlap handling.
        """
        # Calculate volume conversion factor (mm³ to cm³)
        volume_factor = abs(np.prod(self.axis_spacing) / 1000)
        voi_temp = np.zeros_like(self.voi_array)

        for voi_data in self.voi_info:
            roi_num = voi_data["num"]

            # Skip organs not in the needed_organs list
            if not self._is_needed_organ(voi_data, needed_organs):
                continue

            # Reconstruct VOI mask
            voi_mask = self._reconstruct_voi_mask(voi_data)
            voi_data["mask"] = voi_mask

            # Handle overlapping regions if required
            if handling_overlap:
                voi_overlap = voi_temp.copy()
                # Clear overlapping regions in the general VOI array
                voi_temp[voi_mask] = 0
                # Clear overlapping regions in this VOI mask
                voi_mask[(voi_overlap < roi_num) & (voi_overlap != 0)] = 0

            # Add the VOI to the master VOI array
            voi_temp[voi_mask] = roi_num

            # Calculate and store volume information
            num_voi_pixels = np.sum(voi_mask)
            volume = num_voi_pixels * volume_factor
            self.voi_info[roi_num - 1]['volume'] = round(volume, 1)

        # Update the VOI matrix
        self.voi_array = voi_temp.copy()

        # Print debug information if requested
        if self.debug:
            self._print_voi_volume_info(needed_organs, volume_factor)

    def _print_voi_volume_info(self, needed_organs, volume_factor):
        """
        Print volume information for all VOIs.

        Parameters:
            needed_organs (list or str): List of needed organ names or "all".
            volume_factor (float): Conversion factor from voxels to cm³.
        """
        print("Unique values in the 3D mask and their volume:")

        for item in self.voi_info:
            i = item["num"]

            # Skip organs not in the needed_organs list
            if not self._is_needed_organ(item, needed_organs):
                continue

            count = np.count_nonzero(self.voi_array == i)
            volume_cm3 = count * volume_factor
            print(f"{item['name']}, calculated volume: {round(volume_cm3, 1)} cm³")

    def _retrieve_physical_contour_points(self, mask_3d, debug=False, roi_name=None):
        """
        Retrieve physical contour points from the 3D mask of a VOI.

        Parameters:
            mask_3d (np.ndarray): 3D mask array of the VOI.
            debug (bool): Flag to enable debugging output.
            roi_name (str): Name of the ROI for debugging purposes.

        Returns:
            list: List of arrays containing physical contour points for each slice.
        """
        all_physical_contour_points = []

        for slice_idx in range(mask_3d.shape[0]):
            mask = mask_3d[slice_idx].astype(np.uint8)

            # Skip slices with no ROI
            if not np.any(mask):
                continue

            # Get connected components (regions) from the mask
            num_of_regions, label_mask = cv.connectedComponents(mask)

            for region_idx in range(1, num_of_regions + 1):
                region_mask = (label_mask == region_idx).astype(np.uint8)

                # Find contours in the 2D slice image
                contours, _ = cv.findContours(region_mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

                for contour in contours:
                    # Collect all points from contours and add slice index
                    contour_indexes = np.array([[point[0][0], point[0][1], slice_idx] for point in contour])
                    physical_points = self._transform_indexes_to_physical(contour_indexes)

                    if physical_points.size > 0:
                        all_physical_contour_points.append(physical_points.copy())

        return all_physical_contour_points

    def get_organ_mask(self, organ_name):
        """
        Retrieve the mask for a specific organ by name.

        Parameters:
            organ_name (str): Name of the organ to retrieve the mask for.

        Returns:
            np.ndarray or None: The mask of the specified organ if found, otherwise None.
        """
        for organ_data in self.voi_info:
            if organ_data["name"].lower() == organ_name.lower():
                return organ_data["mask"]
        return None

    def add_voi_regions(self, voi_map, voi_names_dict=None, method="update", save_path=None):
        """
        Add new VOIs and save the RT DICOM file.

        Parameters:
            voi_map (np.ndarray): 3D array where each region of interest is labeled by a unique integer.
            voi_names_dict (dict): Dictionary mapping VOI labels to names. Default is empty dict.
            method (str): Method for handling existing VOIs ('update' or 'overwrite').
            save_path (str): Path to save the updated RT DICOM file. Default is None.
        """
        if voi_names_dict is None:
            voi_names_dict = {}

        # Find the maximum VOI region number in the matrix
        max_voi_region = voi_map.max()

        # Prepare to add new ROIs
        new_roi_contour_sequence = []

        for voi_label in range(1, max_voi_region + 1):
            # Get ROI name from dictionary or use default
            roi_name = voi_names_dict.get(str(voi_label), f'tumor {voi_label}')
            roi_number = len(self.voi_info) + 1

            # Create new ROI Structure Set sequence item
            new_roi_sequence = self._create_roi_structure_sequence(roi_number, roi_name)
            self.rt_ds.StructureSetROISequence.append(new_roi_sequence)

            # Choose a unique color for this ROI
            roi_color = self._generate_unique_color()

            # Create new ROI Contour sequence item
            new_roi_contour_sequence_item = self._create_roi_contour_sequence(roi_number, roi_color)

            # Get the VOI mask for this label
            single_voi = voi_map == voi_label

            # Set debug flag based on ROI name
            debug = "tumor" in roi_name

            # Retrieve physical contour points
            all_physical_contour_points = self._retrieve_physical_contour_points(
                mask_3d=single_voi, debug=debug, roi_name=roi_name)

            # Add contour data to the ROI Contour sequence
            for contour in all_physical_contour_points:
                contour_dataset = self._create_contour_dataset(contour)
                new_roi_contour_sequence_item.ContourSequence.append(contour_dataset)

            new_roi_contour_sequence.append(new_roi_contour_sequence_item)

            # Update VOI info in RTSS instance
            self.voi_info.append({
                'num': roi_number,
                'name': roi_name,
                'color': roi_color,
                'contour_points': None,
                'mask': single_voi
            })

        # Add new ROI Contour sequences to the RTSS dataset
        self.rt_ds.ROIContourSequence.extend(new_roi_contour_sequence)

        # Save the updated RTSS file if a path is provided
        if save_path:
            self._save_updated_rtss(save_path)

    def _create_roi_structure_sequence(self, roi_number, roi_name):
        """
        Create a new ROI Structure Set sequence item.

        Parameters:
            roi_number (int): ROI number.
            roi_name (str): ROI name.

        Returns:
            Dataset: New ROI Structure Set sequence item.
        """
        new_roi_sequence = Dataset()
        new_roi_sequence.ROINumber = roi_number
        new_roi_sequence.ReferencedFrameOfReferenceUID = self.rt_ds.ReferencedFrameOfReferenceSequence[
            0].FrameOfReferenceUID
        new_roi_sequence.ROIName = roi_name
        new_roi_sequence.ROIGenerationAlgorithm = "AUTOMATIC"
        return new_roi_sequence

    def _generate_unique_color(self):
        """
        Generate a random color that hasn't been used yet.

        Returns:
            list: RGB color values as a list of 3 integers.
        """
        while True:
            roi_color = [random.randint(0, 255) for _ in range(3)]
            if roi_color not in self.colors_used:
                self.colors_used.append(roi_color)
                return roi_color

    def _create_roi_contour_sequence(self, roi_number, roi_color):
        """
        Create a new ROI Contour sequence item.

        Parameters:
            roi_number (int): ROI number.
            roi_color (list): RGB color values.

        Returns:
            Dataset: New ROI Contour sequence item.
        """
        new_roi_contour_sequence_item = Dataset()
        new_roi_contour_sequence_item.ROIDisplayColor = list(roi_color)
        new_roi_contour_sequence_item.ReferencedROINumber = roi_number
        new_roi_contour_sequence_item.ContourSequence = []
        return new_roi_contour_sequence_item

    def _create_contour_dataset(self, contour):
        """
        Create a new Contour dataset.

        Parameters:
            contour (np.ndarray): Array of contour points.

        Returns:
            Dataset: New Contour dataset.
        """
        contour_dataset = Dataset()
        contour_dataset.ContourGeometricType = "CLOSED_PLANAR"
        contour_dataset.NumberOfContourPoints = len(contour)
        contour_dataset.ContourData = contour.flatten().tolist()  # Transform matrix to list
        return contour_dataset

    def _save_updated_rtss(self, save_path):
        """
        Update RTSS metadata and save to file.

        Parameters:
            save_path (str): Path to save the updated RT DICOM file.
        """
        # Update timestamps
        current_time = datetime.datetime.now()
        self.rt_ds.InstanceCreationDate = current_time.strftime('%Y%m%d')
        self.rt_ds.InstanceCreationTime = current_time.strftime('%H%M%S')
        self.rt_ds.ContentDate = current_time.strftime('%Y%m%d')
        self.rt_ds.ContentTime = current_time.strftime('%H%M%S')

        # Save the updated RTSS file
        self.rt_ds.save_as(save_path)

    def save_mask_and_contour(self, mask_3d, path, roi_name, flag_2d=False, slice_coord=None):
        """
        Save mask and contour visualizations for debugging purposes.

        Parameters:
            mask_3d (np.ndarray): 3D mask array or 2D slice mask.
            path (str): Path to save output images.
            roi_name (str): Name of the ROI.
            flag_2d (bool): Whether the input mask is a 2D slice. Default is False.
            slice_coord (int): Slice coordinate for output filename. Default is None.
        """
        # Ensure the output directory exists
        Path(path).mkdir(parents=True, exist_ok=True)

        if flag_2d:
            self._save_2d_mask_and_contour(mask_3d, path, roi_name)
        else:
            self._save_3d_mask_and_contour(mask_3d, path, roi_name)

    def _save_3d_mask_and_contour(self, mask_3d, path, roi_name):
        """
        Save 3D mask and contour visualization for each slice.

        Parameters:
            mask_3d (np.ndarray): 3D mask array.
            path (str): Path to save output images.
            roi_name (str): Name of the ROI.
        """
        for slice_idx in range(mask_3d.shape[0]):
            mask = mask_3d[slice_idx].astype(np.uint8)

            # Skip slices with no ROI
            if not np.any(mask):
                continue

            # Create RGB image from binary mask
            image_to_save = np.stack((mask * 255, mask * 255, mask * 255), axis=-1)

            # Get connected components (regions) from the mask
            num_of_regions, label_mask = cv.connectedComponents(mask)

            for i in range(1, num_of_regions + 1):
                region_mask = label_mask == i

                # Find contours in the 2D slice image
                contours, _ = cv.findContours(region_mask.astype(np.uint8), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

                # Draw contours on the image in red
                cv.drawContours(image_to_save, contours, -1, (0, 0, 255), 1)

            # Save the image with contours
            cv.imwrite(f"{path}/{roi_name}_slice_{slice_idx}.png", image_to_save)

    def _save_2d_mask_and_contour(self, mask, path, roi_name, slice_coord=None):
        """
        Save 2D mask and contour visualization.

        Parameters:
            mask (np.ndarray): 2D mask array.
            path (str): Path to save output images.
            roi_name (str): Name of the ROI.
            slice_coord (int, optional): Specific slice coordinate. Defaults to None.
        """
        if not np.any(mask):
            return

        # Create RGB image from binary mask
        mask_rgb = np.stack((mask * 255, mask * 255, mask * 255), axis=-1)
        image_to_save = mask_rgb.copy()

        # Get connected components
        num_of_regions, label_mask = cv.connectedComponents(mask)

        for i in range(1, num_of_regions + 1):
            region_mask = (label_mask == i).astype(np.uint8)

            # Find contours
            contours, _ = cv.findContours(region_mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

            # Draw contours on the image in red
            cv.drawContours(image_to_save, contours, -1, (0, 0, 255), 1)

            # Optionally save individual region contours
            for j, contour in enumerate(contours):
                filename = f"{path}/{roi_name}_region_{i}_contour_{j}"
                if slice_coord is not None:
                    filename += f"_slice_{slice_coord}"
                filename += ".png"
                cv.imwrite(filename, cv.drawContours(mask_rgb.copy(), [contour], -1, (0, 0, 255), 1))








