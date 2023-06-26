# -*- coding: utf-8 -*-
# created by Zhang Lizhi, 2023-01-15

import requests
import base64
from PIL import Image
import io
import argparse
import os
import glob


def stringToRGB(base64_string, output_img_fpn):
    imgdata = base64.b64decode((base64_string[22:]))
    img = Image.open(io.BytesIO(imgdata))
    img.save(output_img_fpn)

def main(args):
    """
    """
    vis=args.vis

    imgs = []
    if os.path.isfile(args.input):
        imgs.append(args.input)
    else:
        imgs = glob.glob(os.path.join(args.input, "*" + args.img_format))
        imgs.sort()

    for img_fpn in imgs:
        encoded_string = ""
        with open(img_fpn, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())

        response = requests.post(f"{args.host}/run/predict", json={
            "data": [
                "data:image/png;base64," + encoded_string.decode('utf-8'),
                vis,
            ]}).json()
        data = response["data"]
        if args.vis:
            stringToRGB(data[0], args.vis_dir + "/" + os.path.basename(img_fpn))

        # download csv
        down_url = f"{args.host}/file={data[1]['name']}"
        down_res = requests.get(url=down_url)

        frame_id, _ = os.path.splitext(os.path.basename(img_fpn))
        with open(os.path.join(args.output_dir, f'{frame_id}.json'), "wb") as f:
            f.write(down_res.content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument("--input", required=True, help="image or images directory")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--vis", action="store_true", help="output preticted image")
    parser.add_argument("--host", default="http://172.22.13.50:8101")
    parser.add_argument("--img_format", default=".png")

    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.vis:
        args.vis_dir = f'{args.output_dir}/vis'
        os.makedirs(args.vis_dir, exist_ok=True)

    main(args)
        
