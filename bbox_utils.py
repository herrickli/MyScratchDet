import torch


def point_form(boxes):
    """
    Convert prior boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data
    :param boxes: Center-size default boxes from priorbox layer. (cx, cy, w, h)
    :return: Conver to (xmin, ymin, xmax, ymax) form of boxes
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                     boxes[:, :2] + boxes[:, 2:] / 2), 1)


def intersect(box_a, box_b):
    """
    resize both tensor to (A, B, 2) without malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    :param box_a: (tensor) bounding boxes, Shape: [A,4]
    :param box_b: (tensor) bounding boxes, Shape: [B,4]
    :return: (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """
    Compute the jaccard overlap of two sets of boxes,
    The jaccard overlap is simply the intersection over union of two boxes.
    Here we operate on ground truth boxes and default boxes
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    :param box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
    :param box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    :return: jaccard overlap: (tensor) Shape:[box_a.size(0). box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def encode(matched, priors, variances):
    """
    Encode the variances from the priors box into the ground truth boxes
    we have matched with the prior boxes
    :param matched: (tensor) coords of ground truth for each prior in point-form
                    Shape: [num_priors, 4]
    :param priors: (tensor) prior boxes in center-offset form
                    Shape: [num_priors, 4]
    :param variances:(List[float]) Variances of priors boxes
    :return: (tensor) encoded boxes, Shape: [num_priors, 4]
    """
    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]

    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])

    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]

    # return targe for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)




def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """
    Match each prior box with the ground truth box of the highest jaccard overlap,
    encode the bounding boxes, then return the matched indices corresponding to
    both the confidence and pred locations
    :param threshold: (float) the overlap threshold used when matching boxes
    :param truths: (tensor) Ground Truth boxes. Shape: [num_obj, num_priors]
            num_obj means the number of ground truth boxes in a image.
    :param priors: (tensor) Shape: [num_priors, 4]
    :param variances: (tensor) Variances corresponding to each prior box coord, Shape:[num_priors, 4]
    :param labels: (tensor) All the class labels for the image. Shape:[num_obj]
    :param loc_t: (tensor) Tensor to be filled w/ endecoded location target.
    :param conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
    :param idx: (int) current batch index
    :return: The matched indices corresponding to 1) locations and 2) confidence preds
    """
    # [A, B] A:num_gts, B:num_priors
    overlap = jaccard(truths, point_form(priors))
    # Bipartite Matching
    # [1, num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlap.max(1, keepdim=True)
    # [1, num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlap.max(0, keepdim=True)

    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    best_truth_overlap.index_fill_(0, best_prior_idx, 2) # ensure best prior

    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    matches = truths[best_truth_idx] # shape:  [num_priors, 4]
    conf = labels[best_truth_idx]    # shape:  [num_priors]
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors, 4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior





if __name__ == '__main__':
    a = torch.Tensor([[1, 1, 30, 30], [10, 10, 25, 15]])
    b = torch.Tensor([[10, 10, 20, 20],[10, 10, 20, 20],[10, 10, 20, 20]])
    res = intersect(b, a)
    print(res)