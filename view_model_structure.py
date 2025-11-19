import torch
from model.crnn_model import CnnRnnModel1Channel

# 指定要查看的 checkpoint
CKPT_PATH = r"d:/kwstrainscript/checkpoint/checkpoint_2.2_ft3/crnn_model_best.pth"

# 来自 quantproject_2.2_ft2 的训练配置（与该 checkpoint 一致）
config = {
    "in_c": 16,
    "conv": [
        {"out_c": 16, "k": 8, "s": 2, "p": 1, "dropout": 0.0},
        {"out_c": 32, "k": 4, "s": 2, "p": 1, "dropout": 0.0},
    ],
    "rnn": {"dim": 32, "layers": 1, "dropout": 0.2, "bidirectional": True},
    "fc_out": 2,
}

def main():
    # 实例化并加载权重到 CPU
    model = CnnRnnModel1Channel(config)
    state_dict = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 打印模型结构
    print("=== 模型结构 ===")
    print(model)

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数总数: {total_params}")

    # 关键维度说明
    print("\n=== 关键维度 ===")
    print(f"conv 输出通道: {config['conv'][-1]['out_c']}")
    print(f"GRU: hidden_size={config['rnn']['dim']}, bidirectional={config['rnn']['bidirectional']}, layers={config['rnn']['layers']}")
    print(f"FC 输入维度: {model.fc_in}, 输出维度: {config['fc_out']}")

    # 试跑一次，查看输出维度（输入为 [N, C, T]）
    dummy = torch.randn(1, config["in_c"], 100)
    with torch.no_grad():
        out = model(dummy)
    print("\n=== 试跑输出 ===")
    print(f"输入: {tuple(dummy.shape)} -> 输出: {tuple(out.shape)}")

    # 参数明细
    print("\n=== 参数明细(名称, 形状, 个数) ===")
    for name, p in model.named_parameters():
        print(f"{name:30s} {list(p.shape)!r} {p.numel()}")

if __name__ == "__main__":
    main()