"""
@author: Silvia Zottin
Evaluation code for SAM 2024: few and many shot segmentation ICDAR 2024 competition.

"""

from skimage import io
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
import os
import os.path
import numpy as np


def gen_mask(img):
    img = np.array(img)
    img = img.transpose((2, 0, 1))
    C, H, W = img.shape
    mask = np.zeros([H, W])
    main_text = np.where(
        np.logical_and(
            img[0, :, :] == 255, np.logical_and(img[1, :, :] == 0, img[2, :, :] == 255)
        )
    )
    decoration = np.where(
        np.logical_and(
            img[0, :, :] == 0, np.logical_and(img[1, :, :] == 255, img[2, :, :] == 255)
        )
    )
    title = np.where(
        np.logical_and(
            img[0, :, :] == 255, np.logical_and(img[1, :, :] == 0, img[2, :, :] == 0)
        )
    )
    chapter_headings = np.where(
        np.logical_and(
            img[0, :, :] == 0, np.logical_and(img[1, :, :] == 255, img[2, :, :] == 0)
        )
    )
    paratext = np.where(
        np.logical_and(
            img[0, :, :] == 255, np.logical_and(img[1, :, :] == 255, img[2, :, :] == 0)
        )
    )

    # mask[background] = 0
    mask[main_text] = 1
    mask[decoration] = 2
    mask[title] = 3
    mask[chapter_headings] = 4
    mask[paratext] = 5

    return mask


def udiads_evaluate(result_directory, gt_directory):
    """
    Evaluate the results provided by the files in result_directory with respect
    to the ground truth information given by the files in gt_directory.
    """
    gt_images = []
    result_images = []
    # Check whether result_directory and gt_directory are directories
    if not os.path.isdir(result_directory):
        print("The result folder is not a directory")
        return

    if not os.path.isdir(gt_directory):
        print("The gt folder is not a directory")
        return

    # For each file of the ground truth directory read the result
    for f in sorted(os.listdir(gt_directory)):
        gt_image_path = os.path.join(gt_directory, f)
        gt_image_path = os.path.splitext(gt_image_path)[0] + ".png"
        gt_image = io.imread(gt_image_path)
        gt_image = gen_mask(gt_image)
        gt_images.append(gt_image.flatten())

        result_image_path = os.path.join(result_directory, f)
        result_image_path = os.path.splitext(result_image_path)[0] + ".png"
        result_image = io.imread(result_image_path)
        result_image = gen_mask(result_image)
        result_images.append(result_image.flatten())

    print(f"############## {DS} Scores ##############")
    precision = precision_score(
        np.array(gt_images).flatten(),
        np.array(result_images).flatten(),
        average="macro",
    )
    recall = recall_score(
        np.array(gt_images).flatten(),
        np.array(result_images).flatten(),
        average="macro",
    )
    f1_sc = f1_score(
        np.array(gt_images).flatten(),
        np.array(result_images).flatten(),
        average="macro",
    )
    iou_sc = jaccard_score(
        np.array(gt_images).flatten(),
        np.array(result_images).flatten(),
        average="macro",
    )

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ", f1_sc)
    print("Intersection Over Union: ", iou_sc)
    return iou_sc


result = []
for DS in ["Latin2FS", "Latin14396FS", "Latin16746FS", "Syr341FS"]:

    res = udiads_evaluate(
        result_directory=f"/content/icdar_2024_SAM/Unet-based/{DS}/result",
        gt_directory=f"/content/icdar_2024_SAM/Unet-based/{DS}/gt",
    )
    result.append(res)

print(f"############## Final Scores ##############")
print("Final result of Intersection Over Union: ", np.mean(result))
