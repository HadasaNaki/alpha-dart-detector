import numpy as np
import os
from scipy.ndimage import label, distance_transform_edt
from skimage.filters import threshold_otsu, threshold_multiotsu, threshold_minimum
from SPECT_Data import SPECTData
from RTSS import RTSS
from matplotlib import pyplot as plt
from debug_tools import debug_tumor_segmentation


def find_label_numbers(voi_data, label_names):
    """Find label numbers for specified organ names in a list of volume of interest (VOI) data.

    This function searches for matching organ names in the VOI data and returns their
    corresponding label numbers. The search is case-insensitive and matches partial names.

    Parameters:
    voi_data (list): A list of dictionaries containing VOI data.
                     Each dictionary should have 'name' and 'num' keys.
    label_names (list): A list of strings representing the organ names to search for.

    Returns:
    list: A list of integer label numbers corresponding to the matched organ names.
    """

    label_numbers = []
    for name in label_names:
        name_lower = name.lower()
        for organ_data in voi_data:
            if  name in organ_data["name"].lower():
                label_numbers.append(int(organ_data["num"]))
    return label_numbers


# the old version. when there is overlapping tumor using only the bigger threshold
def tumor_segmentation(spect_data: SPECTData, voi_data: RTSS, debug=False, output_file_path=r"C:\Users\owner\Downloads\SPECT Project\test_rtss_update\file1"):
    """
    Segments tumor regions within the entire SPECT image and defines tumor and non-tumor regions
    within specified dosimetry organs.

    This function processes SPECT data to identify and segment tumor regions, excluding regions
    with natural high uptake. It uses a threshold-based approach to grow tumor regions and
    refines the segmentation based on the connected regions.

    Parameters:
    spect_data (SPECTData): SPECT data containing the image.
    voi_data (RTSS): VOI data containing ROI contour information.
    debug (bool): If True, saves debug images showing the segmentation process.
    output_file_path (str): Path to save the VOI regions after segmentation.

    Returns:
    np.ndarray: Labeled tumor regions.
    """

    spect_image = spect_data.image.copy()
    final_segmented_tumor = np.zeros_like(spect_image, dtype=bool)



    # to plot histogram
    # case_number = output_file_path[-1]
    # plot_histogram_3d(spect_image, save_path=fr"C:\Users\owner\Downloads\SPECT Project\histogram\{case_number}.png")

    # Exclude regions with natural absorb: "high uptake is observed in the salivary glands, duodenum, and kidneys."
    # "normal tissues receiving the highest radiation dose included the salivary and lacrimal glands and the kidneys"
    # doi: 10.2967/jnumed.119.233411. this is for lu-177 -PSMA. however for lu-177 DOTATATE only in the kidney, gallbladder
    # the gallbladder is not defined by the AI organ definer
    scale_factors = np.roll(voi_data.improve_resolution_factor, shift=1) # from (x,y,z) to (z,x,y)
    volume_factor = abs(np.prod(voi_data.axis_spacing) / 1000) * np.prod(scale_factors)

    natural_absorb_organs = ["kidney"]
    natural_absorb_organs_number = find_label_numbers(voi_data=voi_data.voi_info, label_names=natural_absorb_organs)
    for i in natural_absorb_organs_number:
        mask = voi_data.voi_array == i
        spect_image[reduce_matrix(mask, scale_factors)] = 0

    # defined threshold for find initial tumor regions
    max_count = spect_image.max()
    initial_threshold = 0.35 * max_count
    # in our cases 0.35 seem to be the optimal threshold. bigger threshold miss tumors

    # to test to optimal threshold value
    # print(f"Number of initial tumors: {num_tumors}")

    # other method?

    # try to use ostu- threshold method:
    # find the otsu-threshold value.

    # otsu_threshold = threshold_otsu(spect_data.image.ravel())
    # average_otsu =  spect_image_wo_zoom[spect_image_wo_zoom > otsu_threshold].mean()
    # std_otsu = spect_image_wo_zoom[spect_image_wo_zoom > otsu_threshold].std()
    # print(f"i wanted to compare exprimental threshold: {initial_threshold}, otsu: {otsu_threshold}, mean: {average_otsu}, std: {std_otsu}" )

    # try using other thresholding methods
    # normal thershold seem to be not good: as the pixel distribution in not a gussian
    #z = 3
    #z_score_threshold = spect_image.mean() + z * spect_image.std()

    # Entropy_Threshold
    #rescale_factor = np.max(spect_image) / 255
    #normalized_image = spect_data.image.copy() / rescale_factor
    #entropy_threshold = threshold_minimum(normalized_image.flatten()) * rescale_factor

    # multi-otsu method
    #multi_otsu_thresholds = threshold_multiotsu(normalized_image.flatten(), classes=4)  * rescale_factor
    #print(f"threshold: {initial_threshold} the otsu thresholds: {multi_otsu_thresholds}, the entropy_thrshold: {entropy_threshold}")


    # need to try on more cases?



    # Initial segmentation and labeling
    segmented_tumors = spect_image > initial_threshold
    label_initial_tumors, num_tumors = label(segmented_tumors)

    # updating the regions that have local threshold bigger than the initial threshold
    # in these regions sometimes there are connected, but should be a separate objects
    for i in range(1,num_tumors + 1):
        object_mask = label_initial_tumors == i
        local_threshold = spect_image[object_mask].max() * 0.42
        if local_threshold > initial_threshold:
            temp_object_seg = spect_image > local_threshold
            segmented_tumors[object_mask] = temp_object_seg[object_mask]

    # mapping the objects, get their local threshold and find the descending order of indices
    label_initial_tumors, num_tumors = label(segmented_tumors)

    # Grow each region based on 42% of its maximum intensity
    for i in range(1, num_tumors + 1):
        initial_region_mask = label_initial_tumors == i

        # Growing with a more precise threshold only in the tumor connected region
        max_count_in_tumor = spect_image[initial_region_mask].max()
        growing_threshold = 0.42 * max_count_in_tumor

        # label the tumors using the new threshold
        segmented_growing = spect_image > growing_threshold
        label_growing, num_objects = label(segmented_growing)
        growing_object_label = list(np.unique(label_growing[initial_region_mask]))


        # Identify the new object number in the new labeling using the initial object region
        print(growing_object_label)
        growing_object_label = growing_object_label[0]

        # For debugging
        #pixels = np.count_nonzero(initial_region_mask)
        #initial_volume = pixels * volume_factor
        #print(f"Tumor {i}: Initial volume: {initial_volume:.2f} cm^3")
        #print(f"Tumor {i}: Max count in tumor: {max_count_in_tumor}, Growing threshold: {growing_threshold}")

        # Checking for overlap with other tumors
        growing_region_mask = label_growing == growing_object_label
        overlap_labels = np.unique(label_initial_tumors[growing_region_mask])

        if len(overlap_labels) > 1:
            #print (f"num of reconnecting regions {overlap_labels}")
            # Find the max threshold of the growing connected region
            max_threshold = spect_image[growing_region_mask].max()
            threshold_42_from_max = 0.42 * max_threshold

            if growing_threshold < threshold_42_from_max:
                growing_threshold = threshold_42_from_max

                # Perform new growing segmentation with the bigger growing threshold
                segmented_growing = spect_image > growing_threshold
                label_growing, num_objects = label(segmented_growing)
                unique_labels = list(np.unique(label_growing[initial_region_mask]))
                if growing_threshold > initial_threshold:
                    if 0 in unique_labels:
                        unique_labels.remove(0)
                    # מקרה שבו האוביקט מתחבר לאובייקט עם ערך גבוה יחסית אז בשלב זה להתעלם מאובייקט זה
                    #  הערכים שלו גדולים מ  0.35
                    # אך קטנים מ0.42 של האיבר הסמוך
                    if len(unique_labels) == 0:
                        continue
                growing_object_label = unique_labels[0]
                growing_region_mask = label_growing == growing_object_label

        # Checking the tumor volume
        pixels = np.count_nonzero(growing_region_mask)
        growing_volume = pixels * volume_factor
        #print(f"Tumor {i}: Growing volume: {growing_volume:.2f} cm^3")

        # Choosing only sufficiently large tumors
        if growing_volume > 1 : # (1cm^3)
            final_segmented_tumor[growing_region_mask] = True
        else: # try to increase very small object. according to PVA
            final_segmented_tumor[expand_mask(growing_region_mask,1,1,1,1,1,1)] = True

            # print("vol:",np.count_nonzero(final_segmented_tumor[label_growing == growing_object_label] * volume_factor))


    # sorting the labels in descending order by their mean count value.
    label_tumors,_ = label(final_segmented_tumor) # get the 3d connected component of the segmented tumors
    label_tumors = label_by_mean_count(spect_image=spect_image ,labeled_tumors=label_tumors, volume_factor=volume_factor, debug=True)


    # to produce better resolution voi boundaries
    enlarge_factors = np.roll(voi_data.improve_resolution_factor, shift=1) # from (x,y,z) to (z,x,y)
    if any(x != 1 for x in enlarge_factors):
        label_tumors = enlarge_matrix(label_tumors, enlarge_factors)
        spect_image = enlarge_matrix(spect_image, enlarge_factors)


    # Adding the segmented tumors to the VOI data
    voi_data.add_voi_regions(voi_map=label_tumors)

    # segmenting tumor and non tumor region in dosimetry organs (for now only liver)
    voi_organ_segmentation, new_voi_names = organ_without_tumor_segmentation(spect_image, voi_data, output_file_path=output_file_path, case_number=output_file_path[-1])

    # Adding the tumor and non-tumor regions in dosimetry organ to the VOI
    voi_data.add_voi_regions(voi_map=voi_organ_segmentation, voi_names_dict=new_voi_names, save_path=output_file_path)
    print("Saved")

    # For debugging
    for i in range(1, num_tumors + 1):
        volume = volume_factor * np.count_nonzero(label_tumors == i)
        #print(f"Tumor {i}: volume {volume:.2f} cm^3")

    return spect_image, label_tumors, voi_organ_segmentation

# try to segment every object, even if it is connected to another object  ,with his threshold
def tumor_segmentation_try(spect_data: SPECTData, voi_data: RTSS, debug=False, output_file_path=r"C:\Users\owner\Downloads\SPECT Project\test_rtss_update\file1"):
    """
    Segments tumor regions within the entire SPECT image and defines tumor and non-tumor regions
    within specified dosimetry organs.

    This function processes SPECT data to identify and segment tumor regions, excluding regions
    with natural high uptake. It uses a threshold-based approach to grow tumor regions and
    refines the segmentation based on the connected regions.

    Parameters:
    spect_data (SPECTData): SPECT data containing the image.
    voi_data (RTSS): VOI data containing ROI contour information.
    debug (bool): If True, saves debug images showing the segmentation process.
    output_file_path (str): Path to save the VOI regions after segmentation.

    Returns:
    np.ndarray: Labeled tumor regions.
    """

    spect_image = spect_data.image.copy()
    final_segmented_tumor = np.zeros_like(spect_image)



    # to plot histogram
    # case_number = output_file_path[-1]
    # plot_histogram_3d(spect_image, save_path=fr"C:\Users\owner\Downloads\SPECT Project\histogram\{case_number}.png")

    # Exclude regions with natural absorb: "high uptake is observed in the salivary glands, duodenum, and kidneys."
    # "normal tissues receiving the highest radiation dose included the salivary and lacrimal glands and the kidneys"
    # doi: 10.2967/jnumed.119.233411. this is for lu-177 -PSMA. however for lu-177 DOTATATE only in the kidney, gallbladder
    # the gallbladder is not defined by the AI organ definer

    scale_factors = np.roll(voi_data.improve_resolution_factor, shift=1) # from (x,y,z) to (z,x,y)
    volume_factor = abs(np.prod(voi_data.axis_spacing) / 1000) * np.prod(scale_factors)
    # Volume factor to convert to cm^3
    natural_absorb_organs = ["kidney"]
    natural_absorb_organs_number = find_label_numbers(voi_data=voi_data.voi_info, label_names=natural_absorb_organs)
    for i in natural_absorb_organs_number:
        mask = voi_data.voi_array == i
        spect_image[reduce_matrix(mask, scale_factors)] = 0


    # defined threshold for find initial tumor regions
    max_count = spect_image.max()
    initial_threshold = 0.35 * max_count
    # in our cases 0.35 seem to be the optimal threshold. bigger threshold miss tumors
    # print(f"initial threshold: {initial_threshold}")
    # to test to optimal threshold value
    # print(f"Number of initial tumors: {num_tumors}")

    # other method?

    # try to use ostu- threshold method:
    # find the otsu-threshold value.

    # otsu_threshold = threshold_otsu(spect_data.image.ravel())
    # average_otsu =  spect_image_wo_zoom[spect_image_wo_zoom > otsu_threshold].mean()
    # std_otsu = spect_image_wo_zoom[spect_image_wo_zoom > otsu_threshold].std()
    # print(f"i wanted to compare exprimental threshold: {initial_threshold}, otsu: {otsu_threshold}, mean: {average_otsu}, std: {std_otsu}" )

    # try using other thresholding methods
    # normal thershold seem to be not good: as the pixel distribution in not a gussian
    #z = 3
    #z_score_threshold = spect_image.mean() + z * spect_image.std()

    # Entropy_Threshold
    #rescale_factor = np.max(spect_image) / 255
    #normalized_image = spect_data.image.copy() / rescale_factor
    #entropy_threshold = threshold_minimum(normalized_image.flatten()) * rescale_factor

    # multi-otsu method
    #multi_otsu_thresholds = threshold_multiotsu(normalized_image.flatten(), classes=4)  * rescale_factor
    #print(f"threshold: {initial_threshold} the otsu thresholds: {multi_otsu_thresholds}, the entropy_thrshold: {entropy_threshold}")


    # need to try on more cases?

    # Initial segmentation and labeling
    segmented_tumors = spect_image > initial_threshold
    label_initial_tumors, num_tumors = label(segmented_tumors)

    # updating the regions that have local threshold bigger than the initial threshold
    # in these regions sometimes there are connected, but should be a separate objects
    for i in range(1,num_tumors + 1):
        object_mask = label_initial_tumors == i
        local_threshold = spect_image[object_mask].max() * 0.42
        if local_threshold > initial_threshold:
            temp_object_seg = spect_image > local_threshold
            segmented_tumors[object_mask] = temp_object_seg[object_mask]

    # mapping the objects, get their local threshold and find the descending order of indices
    label_initial_tumors, num_tumors = label(segmented_tumors)
    threshold_list, descending_threshold_indices = get_object_thresholds_and_sorted_indices(image=spect_image, label_map=label_initial_tumors) # inverse the list order without the first element which is the background

    # Grow each region based on 42% of its maximum intensity
    current_index = 1
    # objects_in_overlap = []
    for object_num in descending_threshold_indices:

        # object that was connected to region with higher threshold should not be expanded
        # if object_num in objects_in_overlap:
            # continue

        # object initial mask
        object_initial_mask = label_initial_tumors == object_num
        # Growing with a more precise threshold only in the tumor connected region
        growing_threshold = 0.42 * threshold_list[object_num]

        # for debug and try to find more accurate threshold method
        print (f"label: {object_num}, threshold: {growing_threshold}")
        print(f"otsu: {defined_threshold(spect_image, object_initial_mask, threshold_list[object_num])}")

        # label the tumors using the new threshold only in region that wasn't assign as a tumor yet
        object_growing = spect_image > growing_threshold

        # mapping the connected regions
        temp_label, _ = label(object_growing)

        # finding the object number in the new object mapping
        temp_object_num = np.unique(temp_label[object_initial_mask])[0]

        # checking overlap with other objects
        temp_object_region = temp_label == temp_object_num
        overlap_objects = np.unique(label_initial_tumors[temp_object_region])
        overlap_objects = [x for x in overlap_objects if x!=0 and x!=object_num] # removing the background and initial object number

        # changing the growing threshold
        growing_threshold = (growing_threshold + 0.42 * spect_image[temp_object_region].max())/2
        print(f"new threshold {growing_threshold}")
        # defined overlap region to remove from the growing object

        for overlap_obj_num in overlap_objects:

            # to do: changing growing threshold # if there is an overlap defined more moderate threshold
                # growing_threshold = (growing_threshold + 0.42 * spect_image[overlap].max())/2
                # if growing_threshold > threshold_list[object_num]:
                #    growing_threshold = 0.75 * threshold_list
            # checking if this object was grown already. if was, then take the grown object, else take the initial one
            overlap_obj_index = list(descending_threshold_indices).index(overlap_obj_num) + 1
            if overlap_obj_index < current_index: # was already grown
                overlap_obj_mask = final_segmented_tumor == overlap_obj_index
            else: # wasn't grown
                overlap_obj_mask = label_initial_tumors == overlap_obj_num

            # expand the overlap_mask. to prevent in the next step growing on other object regions
            # other tools that perhaps may help,
            # from scipy.ndimage import distance_transform_edt- כלי למדוד מרחק
            # from scipy.ndimage import binary_dilation - כלי להגדלה

            overlap_obj_mask = adjust_expansion_based_on_position(mask_to_expand=overlap_obj_mask, mask2=object_initial_mask)
            temp_object_region[overlap_obj_mask] = False

        # make sure that the initial region wasn't erase
        temp_object_region[object_initial_mask] = True

        # mapping the connected regions
        label_growing_object, _ = label(temp_object_region)

        # finding the object number in the new object mapping
        new_object_num = np.unique(label_growing_object[object_initial_mask])[0]
        print (f"label: {object_num}, threshold: {growing_threshold}")

        # finding the connected growing region object
        object_growing_mask = label_growing_object == new_object_num

        # Checking the tumor volume
        pixels = np.count_nonzero(object_growing_mask)
        growing_volume = pixels * volume_factor
        print (f"{current_index} tumor volume is: {growing_volume}")

        # labeling the tumor
        # Choosing only sufficiently large tumors
        if growing_volume > 1 : # (1cm^3)
            final_segmented_tumor[object_growing_mask] = current_index
        else: # try to increase very small object. according to PVA
            final_segmented_tumor[expand_mask(object_growing_mask,1,1,1,1,1,1)] = current_index
        current_index += 1

    # sorting the labels in descending order by their mean count value.
    label_tumors = label_by_mean_count(spect_image=spect_image ,labeled_tumors=final_segmented_tumor, volume_factor=volume_factor, debug=True)

    # to produce better resolution voi boundaries
    enlarge_factors = np.roll(voi_data.improve_resolution_factor, shift=1) # from (x,y,z) to (z,x,y)
    if any(x != 1 for x in enlarge_factors):
        label_tumors = enlarge_matrix(label_tumors, enlarge_factors)
        spect_image = enlarge_matrix(spect_image, enlarge_factors)

    # Adding the segmented tumors to the VOI data
    voi_data.add_voi_regions(voi_map=label_tumors)

    # segmenting tumor and non tumor region in dosimetry organs (for now only liver)
    voi_organ_segmentation, new_voi_names = organ_without_tumor_segmentation(spect_image, voi_data, output_file_path=output_file_path, case_number=output_file_path[-1])

    # Adding the tumor and non-tumor regions in dosimetry organ to the VOI
    voi_data.add_voi_regions(voi_map=voi_organ_segmentation, voi_names_dict=new_voi_names, save_path=output_file_path)
    print("Saved")

    # For debugging
    for i in range(1, num_tumors + 1):
        volume = volume_factor * np.count_nonzero(label_tumors == i)
        #print(f"Tumor {i}: volume {volume:.2f} cm^3")

    return spect_image, label_tumors, voi_organ_segmentation



def get_object_thresholds_and_sorted_indices(image, label_map):
    """
    Calculate the maximum intensity thresholds for labeled objects in an image and
    return the thresholds along with the sorted indices of the objects in descending order of their thresholds.

    Parameters:
    ----------
    image : numpy.ndarray
        A 2D or 3D array representing the image data. This contains the intensity values of the pixels or voxels.

    label_map : numpy.ndarray
        A 2D or 3D array of the same shape as 'image' representing the labeled regions (e.g., segmented objects).
        Each labeled object is represented by a unique integer, where 0 represents the background.

    Returns:
    ----------
    object_thresholds : list
        A list where each element is the maximum intensity value (threshold) of the corresponding labeled object.
        The first element corresponds to the background with a value of 0.

    sorted_object_indices : numpy.ndarray
        An array of indices representing the labeled objects, sorted in descending order according to their
        maximum intensity thresholds (excluding the background).

    """

    object_thresholds = [0]  # 0 is the background value

    # Determine and sort the maximum threshold of all objects for the region-growing process
    for i in range(1, label_map.max() + 1):
        max_object_threshold = image[label_map == i].max()
        object_thresholds.append(max_object_threshold)

    sorted_object_indices = np.argsort(object_thresholds)[1:][::-1]  # Inverse the order, excluding the background

    return object_thresholds, sorted_object_indices


def label_by_mean_count(spect_image, labeled_tumors, volume_factor, debug=False):
    """ sorting the tumors by their mean count in descending order
    """
    num_tumors = labeled_tumors.max()
    # sort the order of the labels numbers by the volume:
    mean_count_of_tumors = []
    for i in range(1, num_tumors+1):
        mask = spect_image[labeled_tumors == i]
        count_sum = mask.sum()
        pixels_of_tumors =  np.sum(labeled_tumors == i)
        if pixels_of_tumors != 0:
            mean_count = count_sum / pixels_of_tumors
        else:
            mean_count = 0
        mean_count_of_tumors.append(mean_count)


    sorted_indices = np.argsort(mean_count_of_tumors) + 1 # '+1' is to correct the indices to the labels numbers which don't include the background
    sorted_indices = sorted_indices[::-1] # descending order

    # re-sorting the tumor labels by their volumes in descending order:
    sorted_label_tumors = np.zeros_like(labeled_tumors)
    current_index = 1
    for i in sorted_indices:
        sorted_label_tumors[labeled_tumors == i] = current_index
        current_index += 1
        if debug:
            print(f"tumor {current_index-1} volume: {np.sum(labeled_tumors == i) * volume_factor} ")

    return sorted_label_tumors



def organ_without_tumor_segmentation(spect_image, voi_data: RTSS, dosimetry_organs=["liver", "spleen", "bones"], output_file_path="", case_number=""):
    """
    Segments the organs of interest, differentiating between regions with and without tumors.

    This function processes a given set of VOI (Volume of Interest) data and a corresponding
    image that highlights tumor regions. It creates a new segmentation mask that identifies
    both tumor-involved and tumor-free regions within specified organs.

    Parameters:
    voi_data (RTSS): An instance of the RTSS class containing VOI data.
    tumor_image (np.ndarray): A 3D numpy array indicating tumor regions.
    dosimetry_organs (list): A list of organs for which dosimetry calculations are needed.

    Returns:
    np.ndarray: A 3D numpy array with segmented regions labeled.
    dict: A dictionary of names for the new VOI segments created.
    """
    dosimetry_organs = ["liver"]
    new_voi_names = {}
    voi_organ_segmentation = np.zeros_like(spect_image, dtype=int)
    voi_number = 0

    for organ_name in dosimetry_organs:
        organ_mask = voi_data.get_organ_mask(organ_name=organ_name)
        if organ_mask is None:
            continue
        if organ_name == "liver":
            # plot_histogram_3d(spect_image[organ_mask], save_path=fr"C:\Users\owner\Downloads\SPECT Project\histogram\{case_number}.png")
            # calculating mean count by the one tench from the max count value
            liver_pixels = spect_image[organ_mask].copy()
            mean_count_by_value = get_mean_from_one_tench_of_value(liver_pixels, percent=0.10)
            surround_threshold = 3 * mean_count_by_value

            # mean_count_by_pixel = get_mean_from_one_tench_of_pixel(liver_pixels, ratio_of_pixel=0.1)
            #surround_threshold_pix = 2.5 * mean_count_by_pixel
            #surround_threshold = max (surround_threshold_pix, surround_threshold_val)
            #print(f"max:{liver_pixels.max()}, mean count by val: {mean_count_by_value},by pixel: {mean_count_by_pixel}")
            # calculating mean count by the one tench from the liver pixels

            tumor_image = spect_image > surround_threshold
            non_tumor_image = spect_image <= surround_threshold

        # Identify tumor regions within the organ
        organ_surround_mask = organ_mask & tumor_image

        if not np.any(organ_surround_mask):
            continue

        # Identify tumor-free regions within the organ (in the organ and not (~) tumors)
        organ_without_tumor_mask = organ_mask & ~organ_surround_mask
        # equivalent operation
        organ_without_tumor_mask = organ_mask & non_tumor_image

        # debug
        # print(np.any(organ_surround_mask & organ_without_tumor_mask))
        # save_slices_with_masks(spect_image, organ_surround_mask, organ_without_tumor_mask)

        # defined tumor organ roi
        voi_number += 1
        voi_organ_segmentation[organ_surround_mask] = voi_number
        new_voi_names[str(voi_number)] = "tumors_in_" + organ_name

        # defined non-tumor organ region roi
        voi_number += 1
        voi_organ_segmentation[organ_without_tumor_mask] = voi_number
        new_voi_names[str(voi_number)] = organ_name + "_without_tumor"



    return voi_organ_segmentation, new_voi_names



# TO DO:

# improve identify of initial tumor:
    # use histogram for find the initial first hot points of the tumors (in spect 3 times the background)

# consider another method for identify the final tumor regions:
    # to separate two region that are connecting
    # perhaps using region growing

# how to define the organ_without tumor:
    # using first step /histogram/ region growing ?
    # try using the histogram - but it will give us ap problem, it will influence the Dosimetry.
    # => anyway when choosing threshold method it would affect the dosimetry. because it cut all with a threshold

# problem in the MIM:
    # a. read it, but can't open it in the dosimetry program [it is another problem]
    # however, need to save the voi's for the ct (what are the differences?)
    # b. Some tumors voi are not identified (perhaps they are too small, to be identified) [think to be true)

# does the voi can be updated to the following spect images?
    # perhaps calculating the changing in the center-mas changing of the liver, can help us transform the voi coordinate


# help function to define the in organ segmentation threshold value
# by the mean of all the pixels that are 10% values [give better result] - one_tench_of_value
# or by 10% of the pixels that are smallest - one_tench_of_pixel
def get_mean_from_one_tench_of_value(matrix, percent=0.1):
    # Calculate the 10% threshold value
    threshold = np.max(matrix) * percent
    # Create a boolean mask for pixels below the threshold
    low_pixel_mask = matrix < threshold
    # Calculate the sum and count of the low pixels
    low_pixel_sum = np.sum(matrix[low_pixel_mask].astype(np.float64)) # 'without .astype(np.float64)) there were an overflow
    low_pixel_count = np.count_nonzero(low_pixel_mask)

    # Calculate the average of the low pixels
    low_pixel_average = low_pixel_sum / low_pixel_count if low_pixel_count > 0 else 0
    return int(low_pixel_average)


def get_mean_from_one_tench_of_pixel(matrix, ratio_of_pixel=0.1):
    # Find the indices of the lowest 10% pixel values
    num_pixels = int(matrix.size * ratio_of_pixel)
    low_indices = np.argpartition(matrix.flatten(), num_pixels)[:num_pixels]

    # Calculate the average of the lowest 10%  pixel values
    low_pixel_average = np.mean(matrix.flatten()[low_indices])

    return low_pixel_average


# help function for the tumor segmentation. update their number label, by their volume.
# finally was not used: alex prefer order them by mean_count
def label_by_volume(segmented_tumors, volume_factor):
    """ sorting the tumors by their volume in descending order"
    :param segmented_tumors:
    :param volume_factor:
    :return:
    label the segmented tumors by their volumes in descending order
    """

    label_tumors, num_tumors = label(segmented_tumors)
    # sort the order of the labels numbers by the volume:
    volume_of_tumors = [-volume_factor * np.sum(label_tumors == i) for i in range(1, num_tumors+1)]
    sorted_indices = np.argsort(volume_of_tumors) + 1 # '+1' is to correct the indices to the labels numbers which don't include the background

    # re-sorting the tumor labels by their volumes in descending order:
    sorted_label_tumors = np.zeros_like(label_tumors)
    current_index = 1
    for i in sorted_indices:
        sorted_label_tumors[label_tumors == i] = current_index
        current_index += 1

    return sorted_label_tumors


# help function. to get better resolution of each roi border
def enlarge_matrix(matrix, scale_factors):
    """
    Enlarges a 3D matrix by repeating its values along each axis according to the given scale factors.

    Parameters:
    - matrix: np.ndarray, the original 3D matrix.
    - scale_factors: tuple(int, int, int), the scale factors for each axis (x, y, z).

    Returns:
    - np.ndarray, the enlarged 3D matrix.
    """
    # Repeat the matrix along each axis using the scale factors
    for axis, scale in enumerate(scale_factors):
        matrix = np.repeat(matrix, scale, axis=axis)

    return matrix

# help function: for to adjust the organ mask to the spect image. in order to gain program run time.
def reduce_matrix(matrix, scale_factors):
    """
    Reduces a 3D matrix by taking strides along each axis according to the given scale factors.

    Parameters:
    - matrix: np.ndarray, the original 3D matrix.
    - scale_factors: tuple(int, int, int), the scale factors for each axis (x, y, z).

    Returns:
    - np.ndarray, the reduced 3D matrix.
    """
    slices = tuple(slice(None, None, scale) for scale in scale_factors)
    reduced_matrix = matrix[slices]

    return reduced_matrix


# try to define threshold by a bounding box histogram
def defined_threshold(spect_image, mask, max_val, enlarge_factor=5):
    """
    Define a threshold to segment tumor from background based on a bounding box around the core tumor mask.
    :param spect_image: The 3D SPECT image.
    :param mask: 3D binary mask representing the core region (seed).
    :param max_val: Maximum voxel value within the tumor.
    :param enlarge_factor: The number of pixels to enlarge the bounding box.
    :return: Calculated threshold.
    """
    bounding_box = find_bounding_box(mask)
    top, bottom, front, back, left , right = bounding_box.values()

    # creating a bigger box around the roi
    left = left - enlarge_factor if left - enlarge_factor >= 0 else 0
    top = top - enlarge_factor if top - enlarge_factor >= 0 else 0
    front = front - enlarge_factor if front - enlarge_factor >= 0 else 0

    n, m, p = mask.shape
    enlarge_factor += 1 # adjust the factor for slicing
    right = right + enlarge_factor if right + enlarge_factor <= p else p
    bottom = bottom + enlarge_factor if bottom + enlarge_factor <= n else n
    back = back + enlarge_factor if back + enlarge_factor <= m else m

    length = (right - left) * (bottom - top) * (back - front)
    # taking only the bounding region values that are small then the roi max value
    histogram_values = spect_image[top:bottom, front:back, left:right, ]
    histogram_values = histogram_values[histogram_values <= max_val]
    return threshold_otsu(histogram_values.ravel())



def defined_threshold(spect_image, mask, max_val, enlarge_factor=5, weighted=True):
    """
    Define a threshold to segment tumor from background based on a bounding box around the mask.

    :param spect_image: The 3D SPECT image.
    :param mask: 3D binary mask representing the core region (seed).
    :param max_val: Maximum voxel value within the tumor.
    :param enlarge_factor: The number of pixels to enlarge the bounding box.
    :param weighted: Whether to use weighted histogram calculation based on distance from the core region.
    :return: Calculated threshold.
    """
    bounding_box = find_bounding_box(mask)
    top, bottom, front, back, left, right = bounding_box.values()

    # Creating a bigger box around the ROI
    left = left - enlarge_factor if left - enlarge_factor >= 0 else 0
    top = top - enlarge_factor if top - enlarge_factor >= 0 else 0
    front = front - enlarge_factor if front - enlarge_factor >= 0 else 0

    n, m, p = mask.shape
    enlarge_factor += 1  # adjust the factor for slicing
    right = right + enlarge_factor if right + enlarge_factor <= p else p
    bottom = bottom + enlarge_factor if bottom + enlarge_factor <= n else n
    back = back + enlarge_factor if back + enlarge_factor <= m else m

    # Extract the region of interest from the SPECT image
    roi_spect = spect_image[top:bottom, front:back, left:right]

    # Extract corresponding region from the mask (core region)
    roi_mask = mask[top:bottom, front:back, left:right]

    # Mask the SPECT values using max_val
    histogram_values = roi_spect[roi_spect <= max_val]

    # Calculate distance from the core region (where mask is True)
    distances = distance_transform_edt(~roi_mask)

     # Try normalizing differently to avoid small weights
    distances = distances / np.max(distances)  # Normalize distances to [0, 1]

    # Modify the distance weights with an exponential decay or a shifted function
    weights = np.exp(-distances)  # Exponential decay for weights

    # Print out some of the weight values for debugging
    print(f"Min weight: {weights.min()}, Max weight: {weights.max()}")

    # Apply weighting to the histogram values (if weighted)
    if weighted:
        histogram_values_flat = roi_spect.ravel()
        weighted_histogram_values, _ = np.histogram(histogram_values_flat, bins=256, weights=weights.ravel())

        # Print histogram values to ensure they're not too small
        print(f"Weighted histogram values max: {np.max(weighted_histogram_values)}, min: {np.min(weighted_histogram_values)}")

        threshold = threshold_otsu(weighted_histogram_values)
    else:
        threshold = threshold_otsu(histogram_values.ravel())

    print(f"Computed threshold: {threshold}")
    return threshold

# try to create an realtsitc msak to prevent overlap by other ROI
def adjust_expansion_based_on_position(mask_to_expand, mask2):
    """
    Expands mask1 based on its relative position to mask2.
    The expansion is determined based on the relative location of mask1 with respect to mask2.

    Args:
    - mask_to_expand (np.ndarray): The original mask A (3D array with True/False values).
    - mask2 (np.ndarray): The mask B to compare against.

    Returns:
    - np.ndarray: The expanded mask A.
    """

    # Use the function to get the relative position of the two masks
    relative_positions = relative_location(mask_to_expand, mask2)

    # Base and large expansion values
    base_expansion, large_expansion = 4, 10

    # List of directions (keys)
    directions = ['top', 'bottom', 'front', 'back', 'left', 'right']

    # Initialize all directions to base expansion
    expansions = {direction: base_expansion for direction in directions}

    # Axis mapping for relative position
    axis_map = {
        'top/bottom': ('top', 'bottom'),
        'front/back': ('front', 'back'),
        'left/right': ('left', 'right')
    }

    # Adjust expansions based on relative position
    for axis, (negative_dir, positive_dir) in axis_map.items():
        if relative_positions[axis] == negative_dir:
            expansions[negative_dir] = large_expansion
        elif relative_positions[axis] == positive_dir:
            expansions[positive_dir] = large_expansion

    # Expand mask using the computed expansions
    expanded_mask = expand_mask(mask_to_expand,
                                 expansions['top'], expansions['bottom'],
                                 expansions['front'], expansions['back'],
                                 expansions['left'], expansions['right'])

    return expanded_mask

def expand_mask(mask, top, bottom, front, back, left, right):
    """
    Expands a 3D mask by specified numbers of pixels in each dimension (slice, row, col) with asymmetric options.

    Args:
    - mask (np.ndarray): The original mask (a 3D array with True/False values).
    - mask_b
    - left (int): The number of pixels to expand to the left along the columns (col).
    - right (int): The number of pixels to expand to the right along the columns (col).
    - top (int): The number of pixels to expand upwards along the slices (slice).
    - bottom (int): The number of pixels to expand downwards along the slices (slice).
    - front (int): The number of pixels to expand forwards along the rows (row).
    - back (int): The number of pixels to expand backwards along the rows (row).

    Returns:
    - np.ndarray: The expanded mask.
    """
    # Create a copy of the original mask to avoid modifying it
    expanded_mask = mask.copy()

    # Get the shape of the original mask
    shape_slice, shape_row, shape_col = mask.shape


    # Expand along the columns (col)
    if left > 0:
        if shape_col - left > 0:
            expanded_mask[:, :, left:] |= mask[:, :, :-left]
    if right > 0:
        if shape_col + right <= shape_col:
            expanded_mask[:, :, :-right] |= mask[:, :, right:]

    # Expand along the slices (slice)
    if top > 0:
        if shape_slice - top > 0:
            expanded_mask[top:, :, :] |= mask[:-top, :, :]
    if bottom > 0:
        if shape_slice + bottom <= shape_slice:
            expanded_mask[:-bottom, :, :] |= mask[bottom:, :, :]

    # Expand along the rows (row)
    if front > 0:
        if shape_row - front > 0:
            expanded_mask[:, front:, :] |= mask[:, :-front, :]
    if back > 0:
        if shape_row + back <= shape_row:
            expanded_mask[:, :-back, :] |= mask[:, back:, :]

    return expanded_mask

def find_bounding_box(mask):
    """
    Find the bounding box of a 3D object in a boolean mask.

    Parameters:
    mask (numpy.ndarray): A 3D boolean numpy array where True represents the object.

    Returns:
    dict: A dictionary containing the min/max values for each axis (slice, row, col).
    """
    # Check if the mask has any True values
    if not np.any(mask):
        raise ValueError("The mask does not contain any objects (all values are False).")

    # Get indices where the mask is True
    coords = np.argwhere(mask)

    # Extract the minimum and maximum indices for each axis
    min_slice, min_row, min_col = np.min(coords, axis=0)
    max_slice, max_row, max_col = np.max(coords, axis=0)

    # Return the bounding box as a dictionary
    return {
        'top': min_slice,
        'bottom': max_slice,
        'front': min_row,
        'back': max_row,
        'left': min_col,
        'right': max_col
    }

def relative_location(mask1, mask2):
    """
    Determines the relative location of one 3D object compared to another
    based on their boolean masks.

    Parameters:
    mask1 (numpy.ndarray): Boolean mask of the first object (3D array).
    mask2 (numpy.ndarray): Boolean mask of the second object (3D array).

    Returns:
    dict: A dictionary containing the relative location information for
          each axis ('left/right', 'top/bottom', 'front/back').
    """

    # Use the existing find_bounding_box function to get bbox for both masks
    bbox1 = find_bounding_box(mask1)
    bbox2 = find_bounding_box(mask2)

    # Compute axis lengths for bbox1 (object 1)
    length_slice = bbox1['bottom'] - bbox1['top']  # Z-axis (slice)
    length_row = bbox1['back'] - bbox1['front']    # X-axis (row)
    length_col = bbox1['right'] - bbox1['left']    # Y-axis (col)

    # helper function - that get the relative position of other ROI. and according to this data
    def get_relative_position(d1_min, d1_max, d2_min, d2_max, axis_length):
        """
        Compares two bounding box edges on one axis and determines if the
        second object is to the left/right (or top/bottom, front/back) of
        the first.

        If the difference is smaller than one third of the size of the first
        object's axis, they are considered 'same' (returns 'same').

        Parameters:
        d1_min, d1_max: min and max of the first object's bounding box on the axis.
        d2_min, d2_max: min and max of the second object's bounding box on the axis.
        axis_length: the length of the first object along the axis.

        Returns:
        str: 'left', 'right', 'same', 'top', 'bottom', 'front', or 'back' depending
             on the relative position of object 2 to object 1.
        """
        diff_min = d2_min - d1_min
        diff_max = d2_max - d1_max
        threshold = axis_length / 3

        # If both differences are smaller than the threshold, objects are considered at the same position
        if abs(diff_min) < threshold and abs(diff_max) < threshold:
            return 0
        elif diff_min > 0:
            return 1  # Object 2 is to the right/bottom/back
        else:
            return -1  # Object 2 is to the left/top/front

    # Determine relative positions for each axis
    relative_position = {
        'top/bottom': get_relative_position(bbox1['top'], bbox1['bottom'], bbox2['top'], bbox2['bottom'], length_slice),
        'front/back': get_relative_position(bbox1['front'], bbox1['back'], bbox2['front'], bbox2['back'], length_row),
        'left/right': get_relative_position(bbox1['left'], bbox1['right'], bbox2['left'], bbox2['right'], length_col)
    }

    return relative_position


# for debug
def save_slices_with_masks(spect, mask_tumor, mask_health, output_dir=r"C:\Users\owner\Downloads\SPECT Project\test_rtss_update\organ_segment"):
    """
    Save 2D slices from two 3D mask arrays, alongside a combined image showing both masks in color.

    Each slice is saved only if there are non-zero pixels in at least one of the masks.
    The combined image highlights mask_tumor in red and mask_health in green. The output images
    for each slice are saved as three images:
    1. Tumor mask (grayscale)
    2. Healthy mask (grayscale)
    3. Combined mask visualization (Tumor in red, Healthy in green)

    Parameters:
    ----------
    spect : np.ndarray
        A 3D NumPy array representing the SPECT image (H, W, D).
    mask_tumor : np.ndarray
        A 3D NumPy array representing the tumor mask (binary).
    mask_health : np.ndarray
        A 3D NumPy array representing the healthy tissue mask (binary).
    output_dir : str
        Directory where output images will be saved. Will be created if not existing.

    Returns:
    -------
    None
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_slices = mask_tumor.shape[0]

    for i in range(num_slices):
        # Get current slices of SPECT and masks
        slice_spect =  np.ceil(spect[i, :, :].copy() / spect.max() * 255).astype(int)

        # Create two separate copies of the SPECT slice for tumor and healthy masks
        slice_spect_rgb_a = np.stack([slice_spect] * 3, axis=-1)
        slice_spect_rgb_b = np.stack([slice_spect] * 3, axis=-1)

        slice_a = mask_tumor[i, :, :]
        slice_b = mask_health[i, :, :]

        # Mark tumor and healthy regions on separate SPECT slices
        slice_spect_rgb_a[slice_a] = [255, 0, 0]  # Red for tumor
        slice_spect_rgb_b[slice_b] = [0, 255, 0]  # Green for healthy

        # Skip slice if both masks are empty
        if not np.any(slice_a) and not np.any(slice_b):
            continue

        # Create combined mask image (Red for tumor, Green for healthy)
        combined_image = np.zeros((slice_a.shape[0], slice_a.shape[1], 3), dtype=np.uint8)
        combined_image[:, :, 0] = slice_a * 255  # Red for tumor
        combined_image[:, :, 1] = slice_b * 255  # Green for healthy

        # Plot and save the images
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(slice_spect_rgb_a, cmap='gray')
        axes[0].set_title('Tumor Mask')
        axes[0].axis('off')

        axes[1].imshow(slice_spect_rgb_b, cmap='gray')
        axes[1].set_title('Healthy Mask')
        axes[1].axis('off')

        axes[2].imshow(combined_image)
        axes[2].set_title('Combined Mask')
        axes[2].axis('off')

        # Save the figure as an image file
        plt.savefig(os.path.join(output_dir, f'slice_{i:03d}.png'))
        plt.close(fig)


def tumor_segmentation_fixed_threshold(spect_data: SPECTData, voi_data: RTSS, threshold_ratio=0.35):
    """
    Simple tumor segmentation using a fixed global threshold.

    Parameters:
    - spect_data (SPECTData): The 3D SPECT image.
    - voi_data (RTSS): The RTSS object (for spacing/resolution).
    - threshold_ratio (float): The threshold ratio of max uptake (e.g., 0.35 or 0.41).

    Returns:
    - spect_image (np.ndarray): The original SPECT image (possibly enlarged).
    - labeled_tumors (np.ndarray): Labeled tumor regions above threshold.
    - None: Placeholder to match function interface.
    """
    import numpy as np
    from scipy.ndimage import label

    spect_image = spect_data.image.copy()
    max_val = spect_image.max()
    threshold = threshold_ratio * max_val

    segmented = spect_image > threshold
    labeled_tumors, _ = label(segmented)

    # שימור התאמה לרזולוציה אם צריך
    enlarge_factors = np.roll(voi_data.improve_resolution_factor, shift=1)
    if any(x != 1 for x in enlarge_factors):
        labeled_tumors = enlarge_matrix(labeled_tumors, enlarge_factors)
        spect_image = enlarge_matrix(spect_image, enlarge_factors)

    # לא מוסיפים ל־VOI ולא מבצעים שיפורים
    return spect_image, labeled_tumors, None


