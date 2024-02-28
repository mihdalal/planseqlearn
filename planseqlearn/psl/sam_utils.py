import os

import numpy as np
import json
import torch
from PIL import Image

try:
    # Grounding DINO
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import (
        clean_state_dict,
        get_phrases_from_posmap,
    )

    # segment anything
    from segment_anything import build_sam, build_sam_hq, SamPredictor
except:
    pass
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import (
    clean_state_dict,
    get_phrases_from_posmap,
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_dino(model_config_path, model_checkpoint_path, device):
    """
    Load and configure a Grounding DINO model.

    Args:
        model_config_path (str): Path to the model configuration file.
        model_checkpoint_path (str): Path to the model checkpoint file.
        device (str): Device to load the model on (e.g., "cuda" or "cpu").

    Returns:
        model (torch.nn.Module): The loaded and configured Grounding DINO model.
    """
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    print(load_res)
    _ = model.eval()
    return model


def build_models(
    config_file="Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    grounded_checkpoint="Grounded-Segment-Anything/groundingdino_swint_ogc.pth",
    sam_checkpoint="Grounded-Segment-Anything/sam_vit_h_4b8939.pth",
    sam_hq_checkpoint=None,
    use_sam_hq=False,
    device="cuda",
):
    """
    Build and configure the models for grounding and segmentation.

    Args:
        config_file (str): Path to the model configuration file for Grounding DINO.
        grounded_checkpoint (str): Path to the checkpoint file for the Grounding DINO model.
        sam_checkpoint (str): Path to the checkpoint file for the SAM model.
        sam_hq_checkpoint (str): Path to the checkpoint file for the high-quality SAM model.
        use_sam_hq (bool): Whether to use the high-quality SAM model.
        device (str): Device to load the models on (e.g., "cuda" or "cpu").

    Returns:
        dino (torch.nn.Module): The loaded and configured Grounding DINO model.
        sam (SamPredictor): The loaded and configured SAM model.
    """
    # load model
    dino = load_dino(config_file, grounded_checkpoint, device=device)

    # initialize SAM
    if use_sam_hq:
        sam = SamPredictor(build_sam_hq(checkpoint=sam_hq_checkpoint).to(device))
    else:
        sam = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    return dino, sam


def get_grounding_output(
    model,
    image,
    caption,
    box_threshold,
    text_threshold,
    with_logits=True,
    device="cuda",
):
    """
    Get the grounding output from the model for a given image and caption.

    Args:
        model (torch.nn.Module): The grounding model.
        image (torch.Tensor): The input image as a PyTorch tensor.
        caption (str): The caption to ground.
        box_threshold (float): The threshold for filtering bounding boxes.
        text_threshold (float): The threshold for text segmentation.
        with_logits (bool): Whether to include logits in the output.
        device (str): Device to perform calculations on (e.g., "cuda" or "cpu").

    Returns:
        boxes_filt (torch.Tensor): Filtered bounding boxes.
        pred_phrases (list): List of predicted phrases.
    """
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer
        )
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    """
    Display a segmentation mask on the provided matplotlib axis.

    Args:
        mask (np.ndarray): The segmentation mask as a numpy array.
        ax: The matplotlib axis to display the mask on.
        random_color (bool): Whether to use random colors for the mask.

    Returns:
        None
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    """
    Display a bounding box on the provided matplotlib axis.

    Args:
        box (np.ndarray): The bounding box coordinates as a numpy array.
        ax: The matplotlib axis to display the bounding box on.
        label (str): The label to display alongside the bounding box.

    Returns:
        None
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    """
    Save segmentation mask data and related information to files.

    Args:
        output_dir (str): Directory where the data will be saved.
        mask_list (torch.Tensor): List of segmentation masks as PyTorch tensors.
        box_list (torch.Tensor): List of bounding boxes as PyTorch tensors.
        label_list (list): List of corresponding labels.

    Returns:
        None
    """
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, "mask.jpg"),
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.0,
    )

    json_data = [{"value": value, "label": "background"}]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split("(")
        logit = logit[:-1]  # the last is ')'
        json_data.append(
            {
                "value": value,
                "label": name,
                "logit": float(logit),
                "box": box.numpy().tolist(),
            }
        )
    with open(os.path.join(output_dir, "mask.json"), "w") as f:
        json.dump(json_data, f)


def preprocess_image(img_numpy):
    """
    Preprocess an image represented as a NumPy array.

    Args:
        img_numpy (np.ndarray): Input image as a NumPy array.

    Returns:
        image_pil (PIL.Image.Image): PIL image object.
        image_tensor (torch.Tensor): Normalized image tensor.
    """
    image_pil = Image.fromarray(img_numpy)
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image_tensor


def get_seg_mask(
    img_numpy,
    dino,
    sam,
    text_prompts,
    box_threshold=0.3,
    text_threshold=0.25,
    device="cuda",
    debug=False,
    output_dir="outputs",
):
    """
    Obtain segmented masks using the DINO grounding model and SAM.

    Args:
        img_numpy (np.ndarray): Input image as a NumPy array.
        dino: Grounding DINO model.
        sam: SAM model.
        text_prompts (list): List of text prompts for grounding.
        box_threshold (float): Threshold for filtering bounding boxes.
        text_threshold (float): Threshold for filtering text predictions.
        device (str): Device to run the models on.
        debug (bool): If True, enable debug mode.
        output_dir (str): Directory to save output images and data.

    Returns:
        masks (list of torch.Tensor): List of segmented masks.
        image (np.ndarray): Original image as a NumPy array.
        boxes_filt (torch.Tensor): Filtered bounding boxes.
        pred_phrases (list of str): Predicted phrases from the DINO model.
        img_numpy (np.ndarray): Original image as a NumPy array.
    """
    # Convert text prompt list to . separated string
    text_prompt = ". ".join(text_prompts)

    # Load image
    image_pil, image_tensor = preprocess_image(img_numpy)

    # Run grounding DINO model
    boxes_filt, pred_phrases = get_grounding_output(
        dino, image_tensor, text_prompt, box_threshold, text_threshold, device=device
    )
    image = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
    sam.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = sam.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(
        device
    )

    # Predict masks for each text prompt (outputs in the same order as text prompts)
    masks, _, _ = sam.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )

    if debug:
        # Make output dir
        os.makedirs(output_dir, exist_ok=True)
        # Draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)

        plt.axis("off")
        plt.savefig(
            os.path.join(output_dir, "grounded_sam_output.jpg"),
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.0,
        )
        save_mask_data(output_dir, masks, boxes_filt, pred_phrases)

    return masks, image, boxes_filt, pred_phrases, img_numpy


if __name__ == "__main__":
    # build model
    dino, sam = build_models(
        config_file="../../Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounded_checkpoint="../../Grounded-Segment-Anything/groundingdino_swint_ogc.pth",
        sam_checkpoint="../../Grounded-Segment-Anything/sam_vit_h_4b8939.pth",
        sam_hq_checkpoint=None,
        use_sam_hq=False,
        device="cuda",
    )
    img_numpy = cv2.imread("test_sam.png")

    # # run model
    import time

    now = time.time()
    masks, image, boxes_filt, pred_phrases, img_numpy = get_seg_mask(
        img_numpy,
        dino,
        sam,
        text_prompts=["red can"],
        box_threshold=0.3,
        text_threshold=0.25,
        device="cuda",
        debug=False,
        output_dir="sam_outputs",
    )
