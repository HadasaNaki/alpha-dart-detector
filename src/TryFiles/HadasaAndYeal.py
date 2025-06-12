import os
import nibabel as nib
import pydicom
from tqdm import tqdm
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
from skimage.measure import regionprops
from scipy import ndimage
from skimage.filters import threshold_multiotsu
from skimage import morphology

def process_dicom_to_nifti(dicom_dir, save_path=None):
    """
    טוען סדרת קבצי DICOM, ממיר אותם לפורמט NIfTI ומחלץ פרמטרי תצוגה.
    
    Args:
        dicom_dir: נתיב לתיקייה המכילה קבצי DICOM
        save_path: נתיב לשמירת קובץ ה-NIfTI (אופציונלי)
        
    Returns:
        tuple: (nifti_file_path, window_width, window_level, slope, intercept)
    """
    print(f"מעבד קבצי DICOM מתיקייה: {dicom_dir}")
    
    # חיפוש קבצי DICOM
    dicom_files = []
    for root, _, files in os.walk(dicom_dir):
        for file in files:
            file_path = os.path.join(root, file)
            dicom_files.append(file_path)
    
    print(f"נמצאו {len(dicom_files)} קבצים פוטנציאליים")
    
    # טעינת קבצי DICOM
    ct_slices = []
    for file_path in tqdm(dicom_files, desc="טוען קבצי DICOM"):
        try:
            ds = pydicom.dcmread(file_path, force=True)
            if hasattr(ds, 'Modality') and ds.Modality == 'CT' and hasattr(ds, 'pixel_array'):
                ct_slices.append(ds)
        except Exception as e:
            continue
    
    print(f"נטענו {len(ct_slices)} פרוסות CT")
    
    if len(ct_slices) == 0:
        raise ValueError("לא נמצאו קבצי CT תקינים")
    
    # חילוץ פרמטרי Window/Level ו-Rescale מהפרוסה הראשונה
    window_width = None
    window_level = None
    slope = 1.0      # ערך ברירת מחדל
    intercept = 0.0  # ערך ברירת מחדל
    
    ds = ct_slices[0]  # השתמש בפרוסה הראשונה לחילוץ פרמטרים
    
    # חילוץ פרמטרי Rescale
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        print(f"נמצאו פרמטרי המרה: Rescale Slope={slope}, Rescale Intercept={intercept}")
        print(f"הנוסחה להמרה: HU = Raw_value * {slope} + {intercept}")
    
    # בדיקה אם פרמטרי Window קיימים בתגיות הסטנדרטיות
    if hasattr(ds, 'WindowWidth'):
        window_width = float(ds.WindowWidth) if isinstance(ds.WindowWidth, (int, float)) else float(ds.WindowWidth[0])
        print(f"נמצא Window Width: {window_width}")
    
    if hasattr(ds, 'WindowCenter'):
        window_level = float(ds.WindowCenter) if isinstance(ds.WindowCenter, (int, float)) else float(ds.WindowCenter[0])
        print(f"נמצא Window Level: {window_level}")
    
    # אם פרמטרי Window לא נמצאו בתגיות הסטנדרטיות, ננסה לחפש אותם ישירות
    if window_width is None or window_level is None:
        try:
            if (0x0028, 0x1051) in ds:
                window_width = float(ds[0x0028, 0x1051].value) if isinstance(ds[0x0028, 0x1051].value, (int, float)) else float(ds[0x0028, 0x1051].value[0])
                print(f"נמצא Window Width מתגית (0028,1051): {window_width}")
            
            if (0x0028, 0x1050) in ds:
                window_level = float(ds[0x0028, 0x1050].value) if isinstance(ds[0x0028, 0x1050].value, (int, float)) else float(ds[0x0028, 0x1050].value[0])
                print(f"נמצא Window Level מתגית (0028,1050): {window_level}")
        except Exception as e:
            print(f"שגיאה בחילוץ פרמטרים מתגיות: {e}")
    
    # אם עדיין לא נמצאו פרמטרי Window, נשתמש בערכים שכיחים עבור סריקות CT
    if window_width is None:
        window_width = 400  # ברירת מחדל לרקמות רכות
        print(f"לא נמצא Window Width, משתמש בברירת מחדל: {window_width}")
    
    if window_level is None:
        window_level = 40  # ברירת מחדל לרקמות רכות
        print(f"לא נמצא Window Level, משתמש בברירת מחדל: {window_level}")
    
    # הצגת מידע על טווח Hounsfield Units
    min_possible = intercept
    max_possible = 65535 * slope + intercept
    print(f"טווח אפשרי של Hounsfield Units: {min_possible:.1f} עד {max_possible:.1f}")
    print(f"חלון תצוגה נוכחי: {window_level-window_width/2:.1f} עד {window_level+window_width/2:.1f}")

    # מיון פרוסות לפי מיקום
    try:
        ct_slices.sort(key=lambda s: s.ImagePositionPatient[2])
        print("פרוסות מוינו לפי מיקום")
    except Exception as e:
        print(f"לא ניתן למיין פרוסות לפי מיקום: {e}")
    
    # בניית מערך תלת-ממדי
    img_shape = list(ct_slices[0].pixel_array.shape)
    img_3d = np.zeros([len(ct_slices)] + img_shape)
    
    # חילוץ מרווחי פיקסלים
    pixel_spacing = [float(ct_slices[0].PixelSpacing[0]), 
                     float(ct_slices[0].PixelSpacing[1])]
    
    # חישוב מרווח בין פרוסות
    if len(ct_slices) > 1:
        try:
            slice_spacing = abs(ct_slices[1].ImagePositionPatient[2] - ct_slices[0].ImagePositionPatient[2])
        except:
            slice_spacing = ct_slices[0].SliceThickness
    else:
        slice_spacing = ct_slices[0].SliceThickness
    
    # העתקת נתונים
    for i, ds in enumerate(tqdm(ct_slices, desc="בונה נפח תלת-ממדי")):
        img_3d[i] = ds.pixel_array
    
    # המרה ליחידות Hounsfield רק כעת, אחרי שהגדרנו את img_3d
    img_3d = img_3d * slope + intercept
    print(f"נתונים הומרו ליחידות Hounsfield (מקדם={slope}, מצטבר={intercept})")
    print(f"טווח ערכים לאחר המרה: {img_3d.min():.1f} עד {img_3d.max():.1f} HU")
    
    # יצירת אובייקט NIfTI
    affine = np.eye(4)
    affine[0, 0] = pixel_spacing[0]
    affine[1, 1] = pixel_spacing[1]
    affine[2, 2] = slice_spacing
    nifti_img = nib.Nifti1Image(img_3d, affine)
    
    # שמירת הקובץ
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        nib.save(nifti_img, save_path)
        print(f"קובץ NIfTI נשמר בהצלחה בכתובת: {save_path}")
    
    return save_path, window_width, window_level, slope, intercept
 
    print(f"מעבד קבצי DICOM מתיקייה: {dicom_dir}")
    
    # חיפוש קבצי DICOM
    dicom_files = []
    for root, _, files in os.walk(dicom_dir):
        for file in files:
            file_path = os.path.join(root, file)
            dicom_files.append(file_path)
    
    print(f"נמצאו {len(dicom_files)} קבצים פוטנציאליים")
   
def normalize_nii_with_window(nii_file_path, window_width, window_level, 
                             output_path=None, show_slices=False, 
                             rescale_slope=1.0, rescale_intercept=0.0):
    """
    נרמול קובץ NII לטווח 0-255 עם שימוש בפרמטרים Window Width ו-Window Level
    
    פרמטרים:
    nii_file_path (str): נתיב לקובץ הקלט NII
    window_width (float): רוחב החלון (WW) מה-DICOM
    window_level (float): מרכז החלון (WL) מה-DICOM
    output_path (str, optional): נתיב לשמירת הקובץ המנורמל
    show_slices (bool): האם להציג חתכים
    rescale_slope (float): מקדם המרה (לרוב 1.0)
    rescale_intercept (float): ערך הזזה במרחב ה-HU (לרוב -1024 בסריקות CT)
    
    מחזיר:
    numpy.ndarray: המערך המנורמל בטווח 0-255
    """
    # טעינת קובץ ה-NII
    print(f"טוען קובץ NII: {nii_file_path}")
    nii = nib.load(nii_file_path)
    data = nii.get_fdata()
    
    # מציג מידע על הנתונים
    print(f"מידות התמונה: {data.shape}")
    print(f"טווח ערכים בנתונים: {data.min():.1f} עד {data.max():.1f} HU")
    
    # הסברים על מרחב ה-HU והחלון
    print(f"\nמידע על סקאלת Hounsfield:")
    print(f"- מים: 0 HU")
    print(f"- אוויר: ~ -1000 HU")
    print(f"- עצם צפופה: ~ +1000 HU")
    
    # הוספת יכולת להזיז את חלון התצוגה במרחב ה-HU
    # ניתן להוסיף 1024 אם רוצים להזיז את חלון התצוגה
    # זו למעשה "הזזה שנייה" אחרי ה-Rescale Intercept
    adjust_level = 1024  # הוסף 1024 לנקודת המרכז
    adjusted_window_level = window_level + adjust_level
    
    print(f"\nחלון תצוגה מקורי: מרכז={window_level} HU, רוחב={window_width} HU")
    print(f"חלון תצוגה לאחר התאמה: מרכז={adjusted_window_level} HU (הוספנו {adjust_level})")
    
    # חישוב טווח החלון עם ההזזה
    min_value = adjusted_window_level - (window_width / 2)
    max_value = adjusted_window_level + (window_width / 2)
    print(f"טווח החלון לתצוגה: {min_value:.1f} עד {max_value:.1f} HU")
    
    # נרמול עם שימוש בטווח החלון
    # 1. קיטום ערכים מחוץ לטווח החלון
    data_clipped = np.clip(data, min_value, max_value)
    
    # 2. נרמול לטווח 0-255
    normalized_data = ((data_clipped - min_value) / window_width) * 255
    normalized_data = normalized_data.astype(np.uint8)
    
    print(f"טווח ערכים לאחר נרמול: {normalized_data.min()} עד {normalized_data.max()}")
    
    # הצגת חתכים אם התבקש
    if show_slices:
        # בחירת חתך באמצע הנפח
        middle_slice = data.shape[2] // 2
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title('חתך מקורי')
        plt.imshow(data[:, :, middle_slice], cmap='gray')
        plt.colorbar()
        
        plt.subplot(1, 2, 2)
        plt.title('חתך מנורמל (0-255)')
        plt.imshow(normalized_data[:, :, middle_slice], cmap='gray')
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()
    
    # שמירת הקובץ המנורמל אם התבקש
    if output_path:
        normalized_nii = nib.Nifti1Image(normalized_data, nii.affine, nii.header)
        nib.save(normalized_nii, output_path)
        print(f"קובץ מנורמל נשמר ב: {output_path}")
    
    return normalized_data

def create_needle_mask_with_multi_otsu(nii_file_path, num_classes=3, select_brightest=True, output_path=None, show_results=True):
    """
    יוצר מסיכה בינארית לזיהוי אובייקטים בהירים (כמו מחטים) באמצעות Multi-Otsu Thresholding
    
    פרמטרים:
    nii_file_path (str): נתיב לקובץ ה-NII המנורמל
    num_classes (int): מספר הקבוצות לחלוקה (מומלץ 3 או 4)
    select_brightest (bool): אם True, תבחר רק את הקבוצה הבהירה ביותר. אם False, תבחר את כל הקבוצות מעל הסף הנבחר
    output_path (str): נתיב לשמירת המסיכה הבינארית (אופציונלי)
    show_results (bool): האם להציג את התוצאות כגרף
    
    מחזיר:
    tuple: (binary_mask, thresholds) - המסיכה הבינארית וערכי הסף שנמצאו
    """

    print(f"טוען קובץ NII: {nii_file_path}")
    nii = nib.load(nii_file_path)
    data = nii.get_fdata().astype(np.uint8)  # המרה ל-uint8 אם זה עדיין לא
    
    print(f"מבצע Multi-Otsu Thresholding עם {num_classes} קבוצות...")
    
    # חישוב ערכי הסף עם Multi-Otsu
    thresholds = threshold_multiotsu(data, classes=num_classes)
    print(f"נמצאו ערכי סף: {thresholds}")
    
    # יצירת מסיכה בינארית של האזורים הבהירים ביותר
    if select_brightest:
        # בחירת הקבוצה הבהירה ביותר בלבד
        binary_mask = data > thresholds[-1]
        print(f"נוצרה מסיכה בינארית של האזורים הבהירים ביותר (מעל סף {thresholds[-1]})")
    else:
        # בחירת סף אחר (דוגמה: מעל הסף האמצעי)
        threshold_index = len(thresholds) // 2  # בחירת הסף האמצעי
        binary_mask = data > thresholds[threshold_index]
        print(f"נוצרה מסיכה בינארית של האזורים מעל סף {thresholds[threshold_index]}")
    
    # יצירת תמונה מסווגת (לכל פיקסל מוקצית קבוצה)
    # זה שימושי לויזואליזציה
    regions = np.digitize(data, bins=thresholds)
    
    # הצגת התוצאות
    if show_results:
        middle_slice_idx = data.shape[2] // 2  # חתך אמצעי
        
        plt.figure(figsize=(15, 10))
        
        # הצגת ההיסטוגרמה עם ערכי הסף
        plt.subplot(2, 3, 1)
        plt.hist(data.ravel(), bins=256, density=True)
        for thresh in thresholds:
            plt.axvline(thresh, color='r')
        plt.title('היסטוגרמה וערכי סף')
        plt.xlabel('עוצמת פיקסל')
        plt.ylabel('תדירות')
        
        # הצגת התמונה המקורית
        plt.subplot(2, 3, 2)
        plt.imshow(data[:, :, middle_slice_idx], cmap='gray')
        plt.title('תמונה מקורית')
        plt.axis('off')
        
        # הצגת התמונה המסווגת
        plt.subplot(2, 3, 3)
        plt.imshow(regions[:, :, middle_slice_idx], cmap='viridis')
        plt.title(f'חלוקה ל-{num_classes} קבוצות')
        plt.axis('off')
        
        # הצגת המסיכה הבינארית
        plt.subplot(2, 3, 4)
        plt.imshow(binary_mask[:, :, middle_slice_idx], cmap='gray')
        plt.title('מסיכה בינארית')
        plt.axis('off')
        
        # הצגת המסיכה על התמונה המקורית
        plt.subplot(2, 3, 5)
        overlayed = np.stack([
            binary_mask[:, :, middle_slice_idx].astype(float),  # ערוץ אדום
            data[:, :, middle_slice_idx] / 255.0,              # ערוץ ירוק
            data[:, :, middle_slice_idx] / 255.0               # ערוץ כחול
        ], axis=-1)
        plt.imshow(overlayed)
        plt.title('מסיכה על תמונה מקורית')
        plt.axis('off')
        
        # הצגת התפלגות ערכי הפיקסלים לפי קבוצות
        plt.subplot(2, 3, 6)
        class_names = [f"קבוצה {i+1}" for i in range(num_classes)]
        class_counts = [np.sum(regions == i) for i in range(num_classes)]
        plt.bar(class_names, class_counts)
        plt.title('מספר פיקסלים בכל קבוצה')
        plt.ylabel('מספר פיקסלים')
        
        plt.tight_layout()
        plt.show()
        
        # הצגת חתכים נוספים
        view_slices(data, binary_mask, num_slices=5)
    
    # שמירת המסיכה הבינארית
    if output_path:
        binary_mask_nii = nib.Nifti1Image(binary_mask.astype(np.uint8), nii.affine, nii.header)
        nib.save(binary_mask_nii, output_path)
        print(f"המסיכה הבינארית נשמרה ב: {output_path}")
    
    # החזרת המסיכה הבינארית וערכי הסף
    return binary_mask, thresholds

def view_slices(original_data, binary_mask, num_slices=5):
    """
    מציג מספר חתכים של התמונה המקורית והמסיכה הבינארית
    
    פרמטרים:
    original_data: מערך NumPy של התמונה המקורית
    binary_mask: מערך NumPy של המסיכה הבינארית
    num_slices: מספר החתכים להצגה
    """
    # בחירת אינדקסים שווי מרווח לאורך הממד השלישי
    depth = original_data.shape[2]
    step = depth // (num_slices + 1)
    slice_indices = [step * (i+1) for i in range(num_slices)]
    
    # יצירת תרשים עבור כל חתך
    plt.figure(figsize=(15, 4*num_slices))
    
    for i, idx in enumerate(slice_indices):
        # הצגת התמונה המקורית
        plt.subplot(num_slices, 3, i*3+1)
        plt.imshow(original_data[:, :, idx], cmap='gray')
        plt.title(f'חתך {idx} - מקורי')
        plt.axis('off')
        
        # הצגת המסיכה הבינארית
        plt.subplot(num_slices, 3, i*3+2)
        plt.imshow(binary_mask[:, :, idx], cmap='gray')
        plt.title(f'חתך {idx} - מסיכה בינארית')
        plt.axis('off')
        
        # הצגת המסיכה על התמונה המקורית
        plt.subplot(num_slices, 3, i*3+3)
        overlayed = np.stack([
            binary_mask[:, :, idx].astype(float),  # ערוץ אדום
            original_data[:, :, idx] / 255.0,      # ערוץ ירוק
            original_data[:, :, idx] / 255.0       # ערוץ כחול
        ], axis=-1)
        plt.imshow(overlayed)
        plt.title(f'חתך {idx} - שילוב')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()



# דוגמה לשימוש בפונקציה
if __name__ == "__main__":
    dicom_dir= r"C:\Users\yosef\Downloads\AlphaTauDataSet\2023-12__Studies"
    output_file = r"C:\Users\yosef\Downloads\AlphaTauDataSet\output/nii_1.nii.gz"
    output_file2 = r"C:\Users\yosef\Downloads\AlphaTauDataSet\output/normalized_nii1.nii.gz"
    nii_file, window_width, window_level, slope, intercept = process_dicom_to_nifti(dicom_dir, output_file)
    
    
    normalized_data = normalize_nii_with_window(
        nii_file_path=nii_file,
        window_width=window_width,
        window_level=window_level,  
        output_path=output_file2,
        show_slices=True,
        rescale_slope=slope,
        rescale_intercept=intercept
    )

    # create_needle_mask_with_multi_otsu(
    #     nii_file_path=output_file,
    #     num_classes=3,
    #     select_brightest=True,
    #     output_path=r"C:\Users\yosef\Downloads\AlphaTauDataSet\output/needle_mask_finish.nii.gz",
    #     show_results=True
    # )