import numpy as np
import utilities.external.visualization as vis_util
import cv2


def drawTrackedBBs(image_np, output_data):
    for _id, bb in zip(output_data.tracker_ids, output_data.tracked_bbs):
        # Draw the bounding boxes
        h, w, _ = image_np.shape

        cv2.rectangle(image_np, (int(bb[0] * w), int(bb[1] * h)),
                      (int(bb[2] * w), int(bb[3] * h)),
                      (255, 0, 0), 5)
        cv2.putText(image_np,
                    "ID: " + str(_id),
                    (int(bb[0] * w), int(bb[1] * h + 50)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image_np


def drawDetectedBBs(image_np, output_data, score_thresh=0.1):
    '''
    Draws the bounding boxes on the image ``image_np``
    using the coordinates in the ``output_data``
    and with the ``label_list`` as our subset.
    '''
    if output_data.bbs.size > 0:

        image_np = vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_data.bbs,
            output_data.classes.astype(np.int32),
            output_data.scores,
            output_data.category_index,
            max_boxes_to_draw=300,
            use_normalized_coordinates=True,
            line_thickness=3,
            min_score_thresh=score_thresh)
        image_np = drawTrackedBBs(image_np, output_data)

    return image_np
