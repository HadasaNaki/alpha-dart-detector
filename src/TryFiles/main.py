import os
from loading import load_dicom_and_rtss_this_project, load_dicom, load_rtss
from debug_tools import debug_tumor_segmentation  # Assuming display_3d handles the visualization
from debug_tools import save_colored_contours_with_segmentation
from debug_tools import generate_outputs_from_images
from segmentation_FT import tumor_segmentation_fixed_threshold as tumor_seg_simple
from debug_tools import save_manual_segmentation_images, generate_outputs_from_images
from segmentation_FT import tumor_segmentation as tumor_seg_all
from segmentation_FT import tumor_segmentation_try as tumor_seg_all


def process_case(case_num, base_path, output_base_path, needed_organs, resolution_factor):
    """
    Process a single case: load data, perform segmentation, and save results.
    """
    print(f"Processing case: {case_num}\n{'-' * 20}")

    # Define paths
    case_path = os.path.join(base_path, f"{case_num}_1")
    output_file_path = os.path.join(output_base_path, f"file{case_num}")

    # Load DICOM and VOI data
    spect_data, voi_data = load_dicom_and_rtss_this_project(
        case_path,
        needed_organs=needed_organs,
        improve_resolution_factor=resolution_factor
    )

    print("Data loaded successfully.")

    needed_organs = [voi['name'] for voi in voi_data.voi_info]
    # שלב 1: שמירה של איברים בלבד
    save_colored_contours_with_segmentation(
        voi_data=voi_data,
        spect_image=spect_data.image,
        tumor_mask=None,
        save_path=os.path.join(base_path, f"organs_only/{case_num}"),
        organs_list=needed_organs
    )
    image_dir = os.path.join(base_path, f"organs_only/{case_num}")
    output_name = f"{case_num}organs_only"
    generate_outputs_from_images(
        image_folder=image_dir,
        output_name=output_name,
        gif_duration=300  # 300ms בין פרוסות
    )

    # שלב 2: 35%
    spect_image_35, tumor_label_35, _ = tumor_seg_simple(
        spect_data,
        voi_data.copy_instance(),
        threshold_ratio=0.35
    )
    save_colored_contours_with_segmentation(
        voi_data=voi_data,
        spect_image=spect_image_35,
        tumor_mask=tumor_label_35 > 0,
        save_path=os.path.join(base_path, f"organs_with_tumor_35/{case_num}"),
        organs_list=needed_organs
    )
    image_dir = os.path.join(base_path, f"organs_with_tumor_35/{case_num}")
    output_name = f"{case_num}organs_with_tumor_35"
    generate_outputs_from_images(
        image_folder=image_dir,
        output_name=output_name,
        gif_duration=300  # 300ms בין פרוסות
    )

    # שלב 3: 41%
    spect_image_41, tumor_label_41, _ = tumor_seg_simple(
        spect_data,
        voi_data.copy_instance(),
        threshold_ratio=0.41
    )

    save_colored_contours_with_segmentation(
        voi_data=voi_data,
        spect_image=spect_image_41,
        tumor_mask=tumor_label_41 > 0,
        save_path=os.path.join(base_path, f"organs_with_tumor_41/{case_num}"),
        organs_list=needed_organs
    )

    image_dir = os.path.join(base_path, f"organs_with_tumor_41/{case_num}")
    output_name = f"{case_num}organs_with_tumor_41"
    generate_outputs_from_images(
        image_folder=image_dir,
        output_name=output_name,
        gif_duration=300  # 300ms בין פרוסות
    )

    # יצירת תמונות של סגמנטציה ידנית של הרופא


    ''''# מסלול לשמירת התמונות
    manual_output_dir = os.path.join(base_path, f"manual_tumor/{case_num}")
    os.makedirs(manual_output_dir, exist_ok=True)

    # שמירת תמונות עם האיברים והגידולים הידניים (שמות VOI עם "tumor"/"lesion")
    save_manual_segmentation_images(
        voi_data=voi_data,
        spect_image=spect_data.image,
        save_path=manual_output_dir,
        organs_list=needed_organs
    )

    # יצירת GIF ו-PDF מהתמונות שנשמרו
    generate_outputs_from_images(
        image_folder=manual_output_dir,
        output_name=f"{case_num}_manual_tumor",
        gif_duration=300
    )'''

    print(f"Case {case_num} processed.\n")


def main():
    """
    Main function to process multiple cases.
    """
    # Configurations
    base_path = r"C:\Users\shila\Downloads\SPECT project"
    output_base_path = os.path.join(base_path, "test_rtss_update")
    needed_organs = ["liver", "kidney", "tumor"]
    resolution_factor = (1, 1, 1)
    cases_to_process = range(1, 2)

    # Process each case
    for case_num in cases_to_process:
        process_case(case_num, base_path, output_base_path, needed_organs, resolution_factor)


if __name__ == "__main__":
    main()

