import torch

state = torch.load("jit.pth")
state = {"net."+k:v for k,v in state.items()}
state = {"model": state, "model_ema1": state, "model_ema2": state}

torch.save(state, "checkpoint-last.pth")