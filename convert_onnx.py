import torch
import onnx
import onnxruntime as ort
from model.crnn_model import CnnRnnModel1Channel

def load_state(model, path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        model.load_state_dict(obj["model_state_dict"])
    else:
        model.load_state_dict(obj)
    return model

def main(ckpt_path, onnx_path, frames, batch):
    config = {
        "in_c": 16,
        "conv": [{"out_c": 16, "k": 8, "s": 2, "p": 1, "dropout": 0.0},
                 {"out_c": 32, "k": 4, "s": 2, "p": 1, "dropout": 0.0}],
        "rnn": {"dim": 32, "layers": 1, "dropout": 0.2, "bidirectional": True},
        "fc_out": 2
    }
    model = CnnRnnModel1Channel(config)
    load_state(model, ckpt_path)
    model.eval()
    dummy = torch.randn(batch, config["in_c"], frames)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "time"}, "output": {0: "batch"}}
    )
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    _ = sess.run(["output"], {"input": dummy.detach().cpu().numpy()})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=r"d:/kwstrainscript/checkpoint/checkpoint_2.2_ft3/crnn_model_best.pth")
    parser.add_argument("--onnx", default=r"d:/kwstrainscript/checkpoint/checkpoint_2.2_ft3/crnn_model_best.onnx")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    main(args.ckpt, args.onnx, args.frames, args.batch)
    print(args.onnx)