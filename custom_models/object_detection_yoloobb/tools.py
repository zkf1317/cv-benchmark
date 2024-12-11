import cv2
import numpy as np

class_names = ['plane', 'ship', 'storage tank', 'baseball diamond', 'tennis court',
           'basketball court', 'ground track field', 'harbor', 'bridge', 'large vehicle',
           'small vehicle', 'helicopter', 'roundabout', 'soccer ball field', 'swimming pool']

rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))
red_color = (255, 0, 0)

def draw_detections(original_image, detected_boxes, class_labels):
    """
    Draw detected bounding boxes and labels on the image and display it.

    :param original_image: The input image on which to draw the boxes.
    :param detected_boxes: List of detected RotatedBOX objects.
    :param class_labels: List of class labels.
    """
    for detected_box in detected_boxes:
        box = detected_box.box
        points = cv2.boxPoints(box)

        # Rescale the points back to the original image dimensions
        points[:, 0] = points[:, 0]
        points[:, 1] = points[:, 1]
        points = np.int64(points)

        class_id = detected_box.class_index

        # Draw the bounding box with the color for the class
        color = colors[class_id]
        cv2.polylines(original_image, [points],
                      isClosed=True, color=red_color, thickness=2)

        text = f"{class_labels[class_id]}, {detected_box.score:.2f}"
        # Put the class label text with the same color
        cv2.putText(original_image, text, (points[0][0], points[0][1]),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, red_color, 1)

    return original_image