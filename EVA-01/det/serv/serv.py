# -*- coding: utf-8 -*-
# created by Zhang Lizhi, 2023-01-15

import argparse
import glob
import os
import time
import tqdm
import csv
import cv2
import gradio as gr
import detectron2.data.transforms as T
import numpy as np
import json

from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.config import LazyConfig
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    default_setup,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
# constants
import torch


def setup_cfg(args):
    cfg_py = LazyConfig.load(args.config_file)
    cfg_py = LazyConfig.apply_overrides(cfg_py, args.opts)
    default_setup(cfg_py, args)
    return cfg_py


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=None,
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def save_to_json(json_fpn: str = None, data_dict: dict = None):
    class RoundingFloat(float):
        __repr__ = staticmethod(lambda x: format(x, '.3f'))
        
    json.encoder.c_make_encoder = None
    json.encoder.float = RoundingFloat

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
    # json.dump(data_dict, open(json_fpn, mode="wt"), cls=NumpyEncoder)
    json_str = json.dumps(data_dict, ensure_ascii=False, cls=NumpyEncoder)
    with open(json_fpn, 'w', encoding='utf-8') as json_file:
        json_file.write(json_str)


def convert_to_json(inputs, output_fpn, outputs, header=['frame_id', 'track_id',
        'left_top_x', 'left_top_y', 'right_bottom_x', 'right_bottom_y',
        'class_id', 'class_name', 'score', 'mask']):
    boxes, scores, classes, labels, masks, rle_masks = outputs
    # w, h = inputs['width'], inputs['height']

    xyxys = np.around(boxes.tensor.detach().cpu().numpy(), decimals=3).tolist()
    scores = np.around(scores.detach().cpu().numpy(), decimals=3).tolist()
    labels = [l.split(" ")[0] for l in labels]

    rows = []
    for r in range(len(xyxys)):
        # mask = ",".join(['{:.2f}'.format(m) for m in masks[r].polygons[0].tolist()])
        mask = rle_masks[r]
        mask['counts'] = str(mask['counts'], encoding='utf-8')
        xyxy = xyxys[r]
        row = [-1, -1, xyxy[0], xyxy[1], xyxy[2], xyxy[3], classes[r], labels[r], scores[r], mask]
        row_dict = {}
        for v, k in zip(row, header):
            row_dict[k] = v
        rows.append(row_dict)

    result = {
        "objects":rows
    }

    # with open(output_fpn, 'w') as f:
    #     writer = csv.writer(f, delimiter=",")
    #     writer.writerow(header)
    #     writer.writerows(rows)
    save_to_json(output_fpn, result)

    return output_fpn


def infer_local(model, cfg, logger, args):
    if len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for path in tqdm.tqdm(args.input, disable=not args.output):
        # use PIL, to be consistent with evaluation
        original_image = read_image(path, format="BGR")
        height, width = original_image.shape[:2]
        
        aug = T.Resize(
            cfg.model.backbone.net.img_size
        )
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        if args.fp16:
            image = image.half()
        image = image.cuda()

        inputs = {"image": image, "height": height, "width": width}    
        start_time = time.time()

        outputs = model([inputs])
        predictions = outputs[0]
        # predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        original_image = original_image[:, :, ::-1]
        visualizer = Visualizer(original_image, MetadataCatalog.get('coco_2017_train'))
        cpu_device = torch.device("cpu")
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        if args.output:
            os.makedirs(args.output, exist_ok=True)
            out_filename = os.path.join(args.output, os.path.basename(path))
            vis_output.save(out_filename)
        else:
            cv2.imshow("inferd", vis_output.get_image())

def infer_web(model, cfg, logger, args):
    def inference(original_image, vis):
        height, width = original_image.shape[:2]
        aug = T.Resize(
            cfg.model.backbone.net.img_size
        )
        input_img = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(input_img.astype("float32").transpose(2, 0, 1))
        if args.fp16:
            image = image.half()
        image = image.cuda()
        inputs = {"image": image, "height": height, "width": width}
        outputs = model([inputs])
        predictions = outputs[0]

        img = original_image[:, :, ::-1]
        visualizer = Visualizer(img, MetadataCatalog.get('coco_2017_train'))
        cpu_device = torch.device("cpu")

        out_img = None
        output_fpn = "./serv/temp.json"
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)
                boxes, scores, classes, labels, masks, rle_masks = visualizer.parse_output(predictions=instances)
                if vis:
                    out_img = vis_output.get_image()[:, :, ::-1]

                convert_to_json(inputs, output_fpn, [boxes, scores, classes, labels, masks, rle_masks])

        return out_img, output_fpn

    demo = gr.Interface(
        inference, 
        inputs=[gr.Image(), gr.Checkbox(value=False, label="Visualize image ?")], 
        outputs=["image", "file"],
        title="EVA detection Demo",
        examples=[["./serv/000154.jpg"]]
    )
    demo.launch(server_name="0.0.0.0", server_port=8010, show_api=True, show_tips=True)


if __name__ == "__main__":
    # mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    model = instantiate(cfg.model)
    # model.to(cfg.train.device)
    model = create_ddp_model(model, fp16_compression=True)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    if args.fp16:
        model.half()
    model.to(cfg.train.device)
    model.eval()

    if args.input:
        infer_local(model, cfg, logger, args)
    else:
        infer_web(model, cfg, logger, args)