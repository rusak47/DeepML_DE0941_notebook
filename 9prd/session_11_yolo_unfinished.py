import os
import zipfile
import ultralytics
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def path_workaround():
    ## why this bug hapened!?
    curdir = os.getcwd()
    print(f"curdir {curdir}")
    root = 'data'
    if not curdir.endswith('PyCharmMiscProject'):
        rootpath = '../' + root
    path_dataset = f'{root}/datasets'

    print(f"{dataset_path} exists: {os.path.exists(dataset_path)}")
    return root, path_dataset

root, dataset_path = path_workaround()
if not os.path.exists(dataset_path):
    torch.hub.download_url_to_file('https://ultralytics.com/assets/coco2017val.zip', 'tmp.zip')
    os.makedirs(dataset_path, exist_ok=True)
    with zipfile.ZipFile('tmp.zip', 'r') as zip_ref:
        zip_ref.extractall(dataset_path)
    os.remove('tmp.zip')


# TODO

results = []

if len(results) > 0:
    result = results[0]
    img = Image.open(result.path)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(img))

    # Plot the bounding boxes
    ax = plt.gca()
    for box in result.boxes:
        # Get coordinates and convert to numpy
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Create a rectangle patch
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')

        # Add the rectangle to the plot
        ax.add_patch(rect)

        # Add label and confidence if available
        if box.cls is not None and result.names is not None:
            cls_id = int(box.cls[0].item())
            label = f"{result.names[cls_id]} {box.conf[0].item():.2f}"
            plt.text(x1, y1, label, color='white', fontsize=12,
                     bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.title('YOLO Detection Results')
    plt.tight_layout()
    plt.show()


