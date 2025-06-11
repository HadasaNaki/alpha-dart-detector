import numpy as np
from typing import Tuple, Any


class SPECTData:
    """
    Class for storing and managing SPECT imaging data.

    This class stores important information from SPECT scans and provides organized access to the data.

    Attributes:
        pixel_spacing (np.ndarray): The spacing between pixels in each axis (x, y, z) in mm
        image (np.ndarray): The 3D image data matrix
        image_shape (Tuple[int, int, int]): Dimensions of the image (x, y, z)
        image_position (np.ndarray): Position of the first pixel in space (x, y, z)
        image_orientation (np.ndarray): Orientation of the image in space

    Note: Image axes are x (sagittal), y (coronal), z (axial)
    """

    def __init__(self, dicom_dataset) -> None:
        """
        Initialize a SPECTData object from a DICOM file.

        Parameters:
            dicom_dataset: Object containing the loaded DICOM data

        Raises:
            AttributeError: If required fields are not found in the DICOM file
        """
        # Store reference to source data
        self._dicom_data = dicom_dataset

        # Extract pixel spacing
        self.pixel_spacing = np.array([
            dicom_dataset.PixelSpacing[0],
            dicom_dataset.PixelSpacing[1],
            dicom_dataset.SliceThickness
        ])

        # Image data and its dimensions
        self.image = dicom_dataset.pixel_array
        self.image_shape = self.image.shape

        # Image position and orientation in space
        self.image_position = np.array(self._get_attribute(dicom_dataset, 'ImagePositionPatient'))
        self.image_orientation = np.array(self._get_attribute(dicom_dataset, 'ImageOrientationPatient'))

    def _get_attribute(self, ds, attribute: str) -> Any:
        """
        Retrieve a specified attribute from the DICOM data.

        Searches for the attribute in multiple possible locations according to the DICOM file structure.

        Parameters:
            ds: DICOM data object
            attribute (str): Name of the requested attribute

        Returns:
            The value of the requested attribute

        Raises:
            AttributeError: If the attribute is not found in the DICOM file
        """
        # Check if attribute exists directly in the object
        if hasattr(ds, attribute):
            return getattr(ds, attribute)

        # Search in the detector information sequence, if it exists
        elif 'DetectorInformationSequence' in ds:
            for item in ds.DetectorInformationSequence:
                if hasattr(item, attribute):
                    return getattr(item, attribute)

        # Attribute not found
        raise AttributeError(f"Attribute {attribute} not found in the DICOM file.")

    def get_voxel_coordinates(self, i: int, j: int, k: int) -> np.ndarray:
        """
        Calculate the world space coordinates of a specific voxel.

        Parameters:
            i, j, k (int): Voxel indices in the image matrix

        Returns:
            np.ndarray: NumPy array with the voxel's coordinates in world space (in mm)
        """
        # Apply transformation from image position to space position
        position = self.image_position + np.array([
            i * self.pixel_spacing[0],
            j * self.pixel_spacing[1],
            k * self.pixel_spacing[2]
        ])

        return position

    def get_metadata(self) -> dict:
        """
        Returns a dictionary with important metadata from the DICOM file.

        Returns:
            dict: Dictionary containing important metadata
        """
        metadata = {
            'pixel_spacing': self.pixel_spacing.tolist(),
            'image_shape': self.image_shape,
            'image_position': self.image_position.tolist(),
            'image_orientation': self.image_orientation.tolist(),
        }

        # Add additional metadata if available
        optional_fields = ['PatientID', 'PatientName', 'StudyDate', 'Modality']
        for field in optional_fields:
            try:
                metadata[field] = self._get_attribute(self._dicom_data, field)
            except AttributeError:
                pass

        return metadata