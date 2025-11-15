import argparse, torch
from models_def.tiny_unet import TinyUNet

def main(args):
    model = TinyUNet()
    model.load_state_dict(torch.load(args.pt, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1,3,256,256)
    out = "models/lane_unet.onnx"
    torch.onnx.export(model, dummy, out, opset_version=12)
    print("Saved:", out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pt", default="models/best_lane_unet.pt")
    args = p.parse_args()
    main(args)
