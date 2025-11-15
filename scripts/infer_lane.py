import argparse, torch
from models_def.tiny_unet import TinyUNet
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2

def main(args):
    model = TinyUNet()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    tf = T.Compose([T.Resize((256,256)), T.ToTensor()])
    im = Image.open(args.image).convert("RGB")
    x = tf(im).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        p = torch.sigmoid(out)[0,0].numpy()

    mask = (p > 0.5).astype(np.uint8)*255

    img = np.array(im.resize((256,256)))
    overlay = img.copy()
    overlay[mask==255] = [0,255,0]

    vis = cv2.addWeighted(img,0.7,overlay,0.3,0)

    outpath = args.output
    Image.fromarray(vis).save(outpath)
    print("Saved:", outpath)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--output", default="out_lane.png")
    args = p.parse_args()
    main(args)
