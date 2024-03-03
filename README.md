# L3i++ at SAM: Few-Shot and Many-Shot Layout Segmentation of Ancient Manuscripts

[Subtasks](#subtasks) | [Datasets](#data_format) | [Models](#models) | [Contributors](#contributors)

In this repo, we provide our solution to solve two subtasks of [International Competition on Few-Shot and Many-Shot Layout Segmentation of Ancient Manuscripts](https://ai4ch.uniud.it/udiadscomp/).

## Subtasks

- **Task 1: Few-Shot Layout Segmentation.** Create an effective document layout segmentation system using only three images for each manuscript for training and  ten additional images, with the corresponding ground truth, are provided for validation only.

- **Task 2: Many-Shot Layout Segmentation.** Create a layout segmentation system using 35 images per manuscript along with their corresponding ground truth, divided into training, validation, and test sets.

## <a name="data_format"></a>Datasets

The datasets can be accessed at [here](./U-DIADS-Bib-FS).

## <a name="models"></a>Models

We follow the proposed architecture in Rahai et al. (2023), which is L-U-Net-based architecture with a three-step procedure.
The details of the experiment setups are shown in the report. 

## <a name="contributors"></a>Contributors

- [@honghanhh](https://github.com/honghanhh)
- [@nguyennampfiev](https://github.com/nguyennampfiev)
