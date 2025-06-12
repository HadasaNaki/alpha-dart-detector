#func 1 imports
import numpy as np
import nibabel as nib
from skimage.filters import frangi
from skimage.measure import label
import os
import io
import math


#func 2 import
from scipy.ndimage import label
import math

# func 3 import
from scipy.spatial import ConvexHull
from scipy.ndimage import label
import warnings


#func 4 import
from scipy.ndimage import label
from scikit-learn.decomposition import PCA



#helpers

#save_buffer_as_nii(buffer, 'filtered_output.nii')
def save_buffer_as_nii(buffer, filename):
    buffer.seek(0)  # לוודא קריאה מההתחלה
    with open(filename, 'wb') as f:
        f.write(buffer.getbuffer())
    print(f"Saved NIfTI file to {filename}")


def compute_solidity(component_coords):

    if len(component_coords) < 4:
        return 0.0  # לא ניתן לחשב convex hull לפחות מ-4 נקודות ב-3D
    try:
        hull = ConvexHull(component_coords)
        return len(component_coords) / hull.volume
    except:
        return 0.0  # כשל בחישוב hull, כנראה קבוצה פתולוגית


#funcs
def frangiFilter_accuracy(file_binary_mask):
    #details
    nii = nib.load(file_binary_mask)  # file_object יכול להיות io.BytesIO או open('file.nii.gz', 'rb')
    volume = nii.get_fdata()     # זה ייתן לך את המטריצה התלת־ממדית

    voxel_spacing = nii.header.get_zooms()  # ← מחזיר (sx, sy, sz)
    avg_spacing = np.mean(voxel_spacing)
    target_radius_mm = 0.35  # חצי קוטר
    target_radius_voxels = target_radius_mm / avg_spacing

    #frangi defines
    frangi_result = frangi(
    volume,
    scale_range=(max(1,math.ceil(target_radius_voxels)-2), math.ceil(target_radius_voxels)+2),    # גודל הצינורות שנחפש (בסקאלות של גאוס)
    scale_step=1,          # כמה ננסה בכל פעם (1 = ננסה 1, 2, 3, 4)
    alpha=0.5, beta=0.5, gamma=15  # פרמטרים של הפילטר – משפיעים על רגישות לצורת Tube
    )

    #threshold value
    threshold = 0.2  # את יכולה לשנות את זה בהתאם לפלט שתקבלי
    filtered = (frangi_result > threshold).astype(np.uint8)

    #change to .nii file
    filtered_nii = nib.Nifti1Image(filtered, affine=nii.affine, header=nii.header)
    buffer = io.BytesIO()
    filtered_nii.to_file_map({'image': nib.FileHolder(fileobj=buffer)})
    buffer.seek(0)
    return buffer  # ← זה מה שמחזירים ב־Flask/FastAPI עם media_type="application/gzip"



def pixelSize_accuracy(file_binary_mask):
    # טען את הקובץ
    nii = nib.load(file_binary_mask)
    binary_mask = nii.get_fdata()
    voxel_spacing = nii.header.get_zooms()[:3]

    # פרמטרים של הגליל במ"מ
    cylinder_length_mm = 10.0
    cylinder_diameter_mm = 0.7
    cylinder_radius_mm = cylinder_diameter_mm / 2

    # חשב את נפח הגליל במ"מ^3
    cylinder_volume_mm3 = math.pi * (cylinder_radius_mm**2) * cylinder_length_mm

    # חשב את נפח ווקסל ב-mm^3
    voxel_volume_mm3 = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
    mid_voxel_count = cylinder_volume_mm3 / voxel_volume_mm3

    structure = np.ones((3,3,3), dtype=np.int)  # מבנה קישוריות 26 סמוכים
    labeled_array, num_features = label(binary_mask, structure=structure)
    filtered_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    # עבור כל קבוצה מחוברת, בדוק את גודלה וסנן לפי הגודל
    min_voxels = 0.7 * mid_voxel_count
    max_voxels = 1.3 * mid_voxel_count


    for label_idx in range(1, num_features+1):
        component = (labeled_array == label_idx)
        voxel_count = np.sum(component)
        if min_voxels <= voxel_count <= max_voxels:
            # אם הקבוצה לא גדולה מדי, תשאיר אותה
            filtered_mask[component] = 1

    # צור קובץ NIfTI חדש בזיכרון
    filtered_nii = nib.Nifti1Image(filtered_mask, affine=nii.affine, header=nii.header)
    buffer = io.BytesIO()
    filtered_nii.to_file_map({'image': nib.FileHolder(fileobj=buffer)})
    buffer.seek(0)

    return buffer


def soldit_accuracy(file_binary_mask):

    min_solidity=0.7
    nii = nib.load(file_binary_mask)
    binary_mask = nii.get_fdata()

    # מציאת קבוצות מחוברות ב-26 שכנים
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled_array, num_features = label(binary_mask, structure=structure)
    filtered_mask = np.zeros_like(binary_mask)

    for label_idx in range(1, num_features + 1):
        component = (labeled_array == label_idx)
        coords = np.column_stack(np.nonzero(component))
        solidity = compute_solidity(coords)

        if solidity >= min_solidity:
            filtered_mask[component] = 1

    # צור קובץ NIfTI חדש בזיכרון
    filtered_nii = nib.Nifti1Image(filtered_mask, affine=nii.affine, header=nii.header)
    buffer = io.BytesIO()
    filtered_nii.to_file_map({'image': nib.FileHolder(fileobj=buffer)})
    buffer.seek(0)

    return buffer


def aspect_ratio_accuracy(file_binary_mask):
    target_length_mm=10.0
    target_diameter_mm=0.7
    tolerance=0.2
    # טען את הקובץ
    nii = nib.load(file_binary_mask)
    binary_mask = nii.get_fdata()
    voxel_spacing = nii.header.get_zooms()[:3]
    # מצא רכיבים מחוברים (26-קישוריות)
    structure = np.ones((3,3,3), dtype=np.int8)
    labeled_array, num_features = label(binary_mask, structure=structure)

    filtered_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    for label_idx in range(1, num_features + 1):
        component_coords = np.argwhere(labeled_array == label_idx)
        
        # מציאת ממדים של הקופסה המעטפת
        min_coords = component_coords.min(axis=0)
        max_coords = component_coords.max(axis=0)
        size_voxels = max_coords - min_coords + 1

        # ממדים במ"מ
        size_mm = size_voxels * voxel_spacing

        # נניח שהמימד הארוך הוא האורך (length)
        length_mm = np.max(size_mm)
        diameter_mm = np.min(size_mm)  # בערך רוחב / קוטר
        
        # בדיקת טולרנס (קבלה של ±tolerance)
        length_ok = (target_length_mm * (1 - tolerance)) <= length_mm <= (target_length_mm * (1 + tolerance))
        diameter_ok = (target_diameter_mm * (1 - tolerance)) <= diameter_mm <= (target_diameter_mm * (1 + tolerance))

        if length_ok and diameter_ok:
            # שומר את הקבוצה המתאימה במסכה המסוננת
            filtered_mask[labeled_array == label_idx]=1

     # צור קובץ NIfTI חדש בזיכרון
    filtered_nii = nib.Nifti1Image(filtered_mask, affine=nii.affine, header=nii.header)
    buffer = io.BytesIO()
    filtered_nii.to_file_map({'image': nib.FileHolder(fileobj=buffer)})
    buffer.seek(0)

    return buffer     


def axisLen_accuracy(file_binary_mask):

    min_length_mm=9
    max_length_mm=13
    # טען את הקובץ
    nii = nib.load(file_binary_mask)
    binary_mask = nii.get_fdata()
    voxel_size = np.array(nii.header.get_zooms()[:3])

    structure = np.ones((3,3,3), dtype=np.int)
    labeled_array, num_features = label(binary_mask, structure=structure)

    filtered_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    for label_idx in range(1, num_features + 1):
        component = (labeled_array == label_idx)
        coords = np.argwhere(component)

        if len(coords) < 3:
            continue  # לא מספיק נקודות ל-PCA

        # המרת הקואורדינטות לממדי מילימטר על פי ווקסל
        coords_mm = coords * voxel_size

        # ביצוע PCA
        pca = PCA(n_components=1)
        pca.fit(coords_mm)

        # הפרויקטציה של הנקודות על הציר הראשי (וקטור PCA יחידה)
        projected = pca.transform(coords_mm).flatten()

        length = projected.max() - projected.min()

        # סינון לפי אורך
        if min_length_mm <= length <= max_length_mm:
            filtered_mask[component] = 1

     # צור קובץ NIfTI חדש בזיכרון
    filtered_nii = nib.Nifti1Image(filtered_mask, affine=nii.affine, header=nii.header)
    buffer = io.BytesIO()
    filtered_nii.to_file_map({'image': nib.FileHolder(fileobj=buffer)})
    buffer.seek(0)
    return buffer



