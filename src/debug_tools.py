import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.animation import PillowWriter, FuncAnimation
from matplotlib.patches import Patch
from skimage.color import label2rgb
import os
import cv2
import imageio



def debug_tumor_segmentation(spect_image_3d, tumor_segmentation, voi_info, path="", case=0):
    """
    Debugs tumor segmentation by comparing the segmentation with provided VOIs (volumes of interest).
    Saves relevant slices showing the segmentation and overlap, and calculates Dice coefficients for each tumor.

    Parameters:
    - spect_image_3d (np.ndarray): 3D SPECT image array.
    - tumor_segmentation (np.ndarray): 3D array with segmented tumor regions.
    - voi_info (list of dict): List containing VOI information with 'mask' and 'name' keys.
    - path (str): Path to save output images and dice coefficient file.

    Returns:
    - None
    """
    if path == "":
        path = rf"C:\Users\shila\Downloads\SPECT Project\debug_tumor_segmentation\{case}"
        os.makedirs(path, exist_ok=True)

    # Initialize segmentation mask and VOI mask
    segmentation = tumor_segmentation > 0
    dice_cal = []
    voi_mask = np.zeros_like(tumor_segmentation, dtype=bool)

    # Create 3D mask for all VOIs labeled as tumors and calculate Dice coefficients
    for data in voi_info:
        if "tumor" in data["name"].lower() and any(char.isdigit() for char in data["name"]):
            voi_mask[data["mask"]] = True
            # Calculate Dice coefficient and store result with VOI name
            dice_results = calculate_dice(tumor_segmentation, voi_mask)
            if dice_results:
                for entry in dice_results:
                    entry["voi_name"] = data["name"]
                dice_cal.extend(dice_results)

    # Define True Positive (TP), False Negative (FN), and False Positive (FP) regions
    TP = voi_mask & segmentation
    FN = voi_mask & (~segmentation)
    FP = (~voi_mask) & segmentation

     # Define colors and labels for the legend
    colors = {
        "True Positive": [0, 255, 0, 128],  # Green
        "False Negative": [255, 255, 0, 128],  # Yellow
        "False Positive": [255, 0, 0, 128],  # Red
    }

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M")
    output_dir = os.path.join(path, f"{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save Dice coefficients in a text file
    dice_file_path = os.path.join(output_dir, "dice_coefficients.txt")
    with open(dice_file_path, "w") as dice_file:
        for entry in dice_cal:
            dice_file.write(f"VOI: {entry['voi_name']} | with label: {entry['seg_num']} | Dice: {entry['dice']:.4f}\n")




    # Process and save images for each slice
    max_val = np.max(spect_image_3d)  # Maximum value for normalization
    for i, image_slice in enumerate(spect_image_3d):
        if not np.any(TP[i] | FN[i] | FP[i]):
            continue  # Skip slices with no relevant regions

        # Normalize the 16-bit SPECT image slice to 8-bit for visualization
        normalized_slice = (image_slice / max_val * 255).astype(np.uint8)

        # Create RGBA overlay for TP, FN, FP regions
        overlay = np.zeros((*normalized_slice.shape, 4), dtype=np.uint8)
        overlay[FN[i]] = colors["False Negative"]  # Yellow for FN
        overlay[FP[i]] = colors["False Positive"] # Red for FP
        overlay[TP[i]] = colors["True Positive"] # Green for TP


        # Plot and save the overlayed image
        fig, ax = plt.subplots()
        ax.imshow(normalized_slice, cmap='gray')
        ax.imshow(overlay)
        ax.axis("off")
        plt.title(f"Slice {i}")
        # Create legend manually
        legend_elements = [
            Patch(facecolor="green", edgecolor="green", label="True Positive"),
            Patch(facecolor="yellow", edgecolor="yellow", label="False Negative"),
            Patch(facecolor="red", edgecolor="red", label="False Positive"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize="small", framealpha=0.6)


        # Save the debug image in the output directory
        output_image_path = os.path.join(output_dir, f"slice_{i:03d}.png")
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

def calculate_dice(tumor_segmentation, voi_mask):
    """
    Calculates the Dice coefficient for each overlapping tumor region.

    Parameters:
    - tumor_segmentation (np.ndarray): 3D array with segmented tumor regions.
    - voi_mask (np.ndarray): Boolean mask for the VOI region.


    Returns:
    - list of dict: List of dictionaries with Dice coefficients, VOI names, and segmentation IDs.
    """
    compare_mask = tumor_segmentation.astype('bool') & voi_mask
    if not np.any(compare_mask):
        return None

    overlap_tumors = np.unique(tumor_segmentation[compare_mask])
    dice_results = []
    for num in overlap_tumors:
        tumor_mask = (tumor_segmentation == num)
        intersection = np.sum(tumor_mask & voi_mask)
        union = np.sum(tumor_mask) + np.sum(voi_mask)
        dice = 2 * intersection / union
        dice_results.append({"dice": dice, "seg_num": num})
    return dice_results


def inspect_3d_mask(volume, voi_info, spacing):
    unique_values, counts = np.unique(volume, return_counts=True)
    print("Unique values in the 3D mask and their volume:")
    for value, count in zip(unique_values, counts):
        if value > 0:
            name = voi_info[value]["label"]
            volume_cm3 = count * spacing[0] * spacing[1] * spacing[2] / 1000
            print(f"Value {name}: {round(volume_cm3, 1)} cm^3")



def create_gif_from_bool_matrix(matrix, save_dir, case_number):
    # Check if the directory exists, if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create the full path including the case number
    save_path = os.path.join(save_dir, f'case_{case_number}.gif')

    frames = []

    # Loop over the third dimension (slices) of the matrix
    for i in range(matrix.shape[2]):
        # Get the current slice and convert it to uint8 type (0 or 255)
        slice_ = (matrix[:, :, i].astype(np.uint8)) * 255

        # Convert slice to a 3-channel image
        colored_slice = cv2.cvtColor(slice_, cv2.COLOR_GRAY2BGR)
        slice_swap =np.swapaxes(slice_,axis1=0 , axis2=1)
        # Find the contours (boundaries) of the shapes in the slice
        contours, _ = cv2.findContours(slice_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours in red on the colored slice
        cv2.drawContours(colored_slice, contours, -1, (0, 255, 0), 1)

        # Append the frame to the list of frames
        frames.append(colored_slice)

    # Save the frames as a GIF
    imageio.mimsave(save_path, frames, format='GIF', duration=0.3)


def display_3d_mask(volume, voi_info, spect_data, case_num):
    num_slices, num_rows, num_cols = volume.shape
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    # Create a custom colormap for the VOIs
    colors = np.zeros((len(voi_info) + 1, 3))  # Initialize with an extra slot for background
    for info in voi_info:
        voi_number = info["num"] - 1
        colors[voi_number] = np.array(info['color']) / 255.0  # Normalize RGB values to [0, 1]

    custom_cmap = ListedColormap(colors)
    # Define the colors for the colormap
    colors = ['black', 'white']
    # Create the colormap
    black_to_red_cmap = LinearSegmentedColormap.from_list('black_to_white', colors)

    def update(slice_idx):
        ax.clear()
        img = ax.imshow(spect_data.image[slice_idx, :, :], cmap=black_to_red_cmap)
        ax.imshow(volume[slice_idx, :, :], cmap=custom_cmap, alpha=0.5)

        for info in voi_info:
            voi_number = info["num"]
            if np.any(volume[slice_idx, :, :] == voi_number):
                label_position = np.argwhere(volume[slice_idx, :, :] == voi_number)[0]
                ax.text(label_position[1], label_position[0], info['name'], color='white', fontsize=8, ha='center', va='center')

        ax.set_title(f'Slice {slice_idx}')
        return img,

    anim = FuncAnimation(fig, update, frames=num_slices, blit=False)

    # Adding a colorbar for the intensity values
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=black_to_red_cmap), ax=ax, orientation='vertical')
    cbar.set_label('Intensity')

    # Save as GIF
    anim.save(rf'C:\Users\shila\Downloads\SPECT Project\debug_load_rtss\volume_visualization{case_num}.gif', writer=PillowWriter(fps=6))





def save_segmentation_gif(matrix, background_image, save_dir, case_number):

    # Check if the directory exists, if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create the full path including the case number
    save_path = os.path.join(save_dir, f'case_{case_number}.gif')

    frames = []

    # Ensure the background image has the same dimensions as the matrix slices
    if background_image.shape[:2] != matrix.shape[:2]:
        raise ValueError("Background image dimensions must match the first two dimensions of the matrix.")

    # Loop over the third dimension (slices) of the matrix
    for i in range(matrix.shape[2]):
        # Get the current slice and convert it to uint8 type (0 or 255)
        slice_ = (matrix[:, :, i].astype(np.uint8)) * 255

        # Convert the background image to BGR (if it's not already) and ensure it's 3 channels
        if len(background_image.shape) == 2 or background_image.shape[2] == 1:
            background_bgr = cv2.cvtColor(background_image, cv2.COLOR_GRAY2BGR)
        else:
            background_bgr = background_image.copy()

        # Find the contours (boundaries) of the shapes in the slice
        contours, _ = cv2.findContours(slice_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask from the slice
        mask = cv2.merge([np.zeros_like(slice_), slice_, np.zeros_like(slice_)])

        # Apply the mask to the background to color the matrix in green
        colored_slice = cv2.addWeighted(background_bgr, 1.0, mask, 1.0, 0)

        # Draw the contours in red on the colored slice
        cv2.drawContours(colored_slice, contours, -1, (0, 0, 255), 2)

        # Append the frame to the list of frames
        frames.append(colored_slice)

    # Save the frames as a GIF
    imageio.mimsave(save_path, frames, format='GIF', duration=0.3)





def plot_histogram_3d(matrix, save_path):
  # Flatten the 3D matrix into a 1D array
  flat_matrix = matrix.flatten()
  step = 20
  max_val = 1000
  if matrix.max() > 13000:
      step = 500
      max_val = 13000


  # Define bin edges
  bin_edges = list(range(0, max_val+1, step))
  for val in [2500,5000,10000,100000]:
    if val > max_val:
        bin_edges.append(val)  # Add a bin for values greater than 1000

  # Calculate histogram
  hist, _ = np.histogram(flat_matrix, bins=bin_edges)

  # Calculate percentages
  percentages = hist / flat_matrix.size * 100

  # Create bar plot
  plt.figure(figsize=(20, 8))  # Adjust figure size for better readability
  plt.bar(range(len(hist)), percentages, width=0.8, align='center')

  # Set x-tick labels
  x_tick_labels = [f'{i}-{i+step-1}' for i in range(0, max_val+1, step)]
  for val in [1000,2500,5000,10000,13000]:
      if val > max_val:
        x_tick_labels.append(f'>{val}')

  plt.xticks(range(len(hist)), x_tick_labels, rotation=45, ha='right')

  # Add number of pixels on each bin
  for i, v in enumerate(hist):
    plt.text(i, percentages[i], str(v), ha='center', va='bottom')

  # Set axis labels and title
  plt.xlabel('Value Range')
  plt.ylabel('Percentage')
  plt.title('Histogram of SPECT')

  # Display the plot
  plt.tight_layout()
  plt.savefig(save_path)
  plt.close()



def plot_histogram(image_values, save_path, bins = None):
    if bins is None:
        percentiles_0_30 = image_values.max() * np.array([0, 3, 6, 10, 20, 30]) / 100  # קפיצות של 5% עד 40%
        percentiles_40_100 = image_values.max() * np.array([40, 60, 100]) / 100
        bins = np.concatenate([
        percentiles_0_30,
        percentiles_40_100
                                ])
    hist, bins = np.histogram(image_values, bins=bins)
# חישוב האחוזים עבור כל עמודה
    hist_percent = (hist / hist.sum()) * 100

# יצירת הגרף
    fig, ax = plt.subplots(figsize=(20, 8))

    bars = ax.bar(bins[:-1], hist_percent, width=np.diff(bins), edgecolor='black', align='edge')  # רוחב עמודות בהתאם לגבולות

    # הוספת טקסט על כל עמודה עם מספר הפיקסלים והאחוזים
    for bar, count, percent, bin_start, bin_end in zip(bars, hist, hist_percent, bins[:-1], bins[1:]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{count}\n({percent:.1f}%)',
                ha='center', va='bottom', fontsize=10)  # גודל גופן גדול יותר
        ax.text(bar.get_x() + bar.get_width() / 2, -10,
                f'{bin_start:.0f}-{bin_end:.0f}',
                ha='center', va='top', fontsize=9)  # טווח הערכים



# הגדרות ציר ה-y באחוזים
    ax.set_ylabel('Percentage of pixels (%) ')
    ax.set_xlabel('Pixel Values')
    ax.set_title('Histogram of SPECT Image')
    ax.set_ylim(0, 100)

    plt.savefig(save_path)

def Psave_colored_contours_with_segmentation(voi_data, spect_image, tumor_mask=None, save_path="", organs_list=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.colors import to_rgba
    from skimage import measure

    os.makedirs(save_path, exist_ok=True)
    num_slices = voi_data.voi_array.shape[0]
    max_val = np.max(spect_image)

    # Colormap בהיר (גוונים טובים לרקע כהה)
    cmap = plt.cm.get_cmap("Set2", 12)

    for z in range(num_slices):
        fig, ax = plt.subplots(figsize=(6, 6))

        # רקע ה-SPECT עם הבהרה קלה
        norm_slice = (spect_image[z] / max_val * 255).astype(np.uint8)
        norm_slice = np.clip(norm_slice, 10, 255)
        ax.imshow(norm_slice, cmap='gray')

        drew_any_organ = False
        organs_in_slice = []

        for idx, organ_data in enumerate(voi_data.voi_info):
            if organs_list and organ_data["name"] not in organs_list:
                continue
            mask = organ_data.get("mask", None)
            if mask is None or not np.any(mask[z]):
                continue

            drew_any_organ = True
            organs_in_slice.append(organ_data["name"].replace(" ", "_"))

            """color = base_colors[idx % len(base_colors)]
            rgba = to_rgba(color, alpha=0.35)

            colored_mask = np.zeros((*mask[z].shape, 4))
            colored_mask[mask[z]] = rgba
            ax.imshow(colored_mask)"""

            # צבע שקוף מתוך Set2
            color = cmap(idx % cmap.N)
            rgba = (*color[:3], 0.35)

            colored_mask = np.zeros((*mask[z].shape, 4))
            colored_mask[mask[z]] = rgba
            ax.imshow(colored_mask, interpolation='none')

            """# ציור קו מתאר לבן סביב האיבר
            ax.contour(mask[z], colors='white', linewidths=1)"""

            '''# קו מתאר מדויק בעזרת skimage
            contours = measure.find_contours(mask[z], 0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color='white', linewidth=1)'''

        if not drew_any_organ:
            plt.close()
            continue

        # קו מתאר אדום לגידול
        if tumor_mask is not None and np.any(tumor_mask[z]):
            tumor_contours = measure.find_contours(tumor_mask[z], 0.5)
            for contour in tumor_contours:
                ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1)

        ax.set_title(f"Slice {z}")
        ax.axis('off')

        organs_str = "_".join(organs_in_slice) if organs_in_slice else "no_organ"
        filename = f"slice_{z:03}_{organs_str}.png"
        plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', pad_inches=0)
        plt.close()

def save_colored_contours_with_segmentation(voi_data, spect_image, tumor_mask=None, save_path="", organs_list=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from skimage import measure

    os.makedirs(save_path, exist_ok=True)
    num_slices = voi_data.voi_array.shape[0]
    max_val = np.max(spect_image)

    cmap = plt.cm.get_cmap("tab10", 12)  # צבעים בהירים

    # זיהוי אוטומטי של כל הגידולים הידניים לפי שם ה־VOI
    manual_mask = np.zeros_like(voi_data.voi_array, dtype=bool)
    for voi in voi_data.voi_info:
        name = voi["name"].lower()
        if any(keyword in name for keyword in ["tumor", "lesion", "gtv", "roi", "target"]):
            mask = voi.get("mask", None)
            if mask is not None:
                manual_mask |= mask

    for z in range(num_slices):
        fig, ax = plt.subplots(figsize=(6, 6))

        norm_slice = (spect_image[z] / max_val * 255).astype(np.uint8)
        norm_slice = np.clip(norm_slice, 10, 255)
        ax.imshow(norm_slice, cmap='gray')

        drew_any_organ = False
        organs_in_slice = []

        # ציור האיברים
        for idx, organ_data in enumerate(voi_data.voi_info):
            if organs_list and organ_data["name"] not in organs_list:
                continue
            mask = organ_data.get("mask", None)
            if mask is None or not np.any(mask[z]):
                continue

            drew_any_organ = True
            organs_in_slice.append(organ_data["name"].replace(" ", "_"))

            color = cmap(idx % cmap.N)
            rgba = (*color[:3], 0.3)
            colored_mask = np.zeros((*mask[z].shape, 4))
            colored_mask[mask[z]] = rgba
            ax.imshow(colored_mask, interpolation='none')

            contours = measure.find_contours(mask[z], 0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color='white', linewidth=1)

        if (
            not drew_any_organ and
            (tumor_mask is None or not np.any(tumor_mask[z])) and
            (manual_mask is None or not np.any(manual_mask[z]))
        ):
            plt.close()
            continue

        # קו כחול – גידול ידני
        if manual_mask is not None and np.any(manual_mask[z]):
            contours = measure.find_contours(manual_mask[z], 0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color='blue', linewidth=1)

        # קו אדום – סגמנטציה אוטומטית
        if tumor_mask is not None and np.any(tumor_mask[z]):
            contours = measure.find_contours(tumor_mask[z], 0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1)

        # Legend (מקרא)
        legend_elements = [
            Line2D([0], [0], color='white', lw=1, label='Organ boundary'),
            Line2D([0], [0], color='blue', lw=2, label='Manual tumor'),
            Line2D([0], [0], color='red', lw=2, label='Auto segmentation'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize='small', framealpha=0.7)

        ax.set_title(f"Slice {z}")
        ax.axis('off')

        organs_str = "_".join(organs_in_slice) if organs_in_slice else "no_organ"
        filename = f"slice_{z:03}_{organs_str}.png"
        plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', pad_inches=0)
        plt.close()





def generate_outputs_from_images(image_folder, output_name, output_dir=None, gif_duration=300):
    import os
    import imageio.v2 as imageio
    from PIL import Image

    if output_dir is None:
        output_dir = image_folder

    # אסוף את כל קבצי ה-PNG מהמיין, ממוינים לפי שם
    images = sorted([
        os.path.join(image_folder, fname)
        for fname in os.listdir(image_folder)
        if fname.lower().endswith(".png")
    ])

    if not images:
        print(f"⚠️ No images found in {image_folder}")
        return

    # 📄 יצירת PDF
    pdf_path = os.path.join(output_dir, f"{output_name}.pdf")
    pil_images = [Image.open(img).convert("RGB") for img in images]
    pil_images[0].save(pdf_path, save_all=True, append_images=pil_images[1:])
    print(f"📄 PDF saved to: {pdf_path}")

    # 🌀 יצירת GIF
    gif_path = os.path.join(output_dir, f"{output_name}.gif")
    frames = [imageio.imread(img) for img in images]
    imageio.mimsave(gif_path, frames, duration=gif_duration / 1000.0)
    print(f"🌀 GIF saved to: {gif_path}")

def save_manual_segmentation_images(voi_data, spect_image, save_path="", organs_list=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.colors import to_rgba
    from skimage import measure

    os.makedirs(save_path, exist_ok=True)
    num_slices = spect_image.shape[0]
    max_val = np.max(spect_image)

    cmap = plt.cm.get_cmap("Set2", 12)

    # יצירת מסכה אחת לכל הגידולים שסומנו ידנית
    manual_mask = np.zeros_like(voi_data.voi_array, dtype=bool)
    for voi in voi_data.voi_info:
        name = voi["name"].lower()
        if any(keyword in name for keyword in ["tumor", "lesion", "gtv", "roi", "target"]):
            mask = voi.get("mask", None)
            if mask is not None:
                print(f"✅ Including manual VOI: {voi['name']}")
                manual_mask |= mask

    for z in range(num_slices):
        fig, ax = plt.subplots(figsize=(6, 6))

        norm_slice = (spect_image[z] / max_val * 255).astype(np.uint8)
        norm_slice = np.clip(norm_slice, 10, 255)
        ax.imshow(norm_slice, cmap='gray')

        drew_any_organ = False
        organs_in_slice = []

        for idx, organ_data in enumerate(voi_data.voi_info):
            if organs_list and organ_data["name"] not in organs_list:
                continue
            mask = organ_data.get("mask", None)
            if mask is None or not np.any(mask[z]):
                continue

            drew_any_organ = True
            organs_in_slice.append(organ_data["name"].replace(" ", "_"))

            color = cmap(idx % cmap.N)
            rgba = (*color[:3], 0.35)

            colored_mask = np.zeros((*mask[z].shape, 4))
            colored_mask[mask[z]] = rgba
            ax.imshow(colored_mask, interpolation='none')

            contours = measure.find_contours(mask[z], 0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color='white', linewidth=1)

        if not drew_any_organ and not np.any(manual_mask[z]):
            plt.close()
            continue

        # קו מתאר כחול לכל הגידולים הידניים
        if np.any(manual_mask[z]):
            tumor_contours = measure.find_contours(manual_mask[z], 0.5)
            for contour in tumor_contours:
                ax.plot(contour[:, 1], contour[:, 0], color='blue', linewidth=2)

        ax.set_title(f"Slice {z}")
        ax.axis('off')

        organs_str = "_".join(organs_in_slice) if organs_in_slice else "no_organ"
        filename = f"slice_{z:03}_{organs_str}.png"
        plt.savefig(os.path.join(save_path, filename), bbox_inches='tight', pad_inches=0)
        plt.close()






