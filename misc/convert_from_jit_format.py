import torch

state = torch.load("checkpoint-last.pth")
state = state["model_ema1"]
state = {k.replace("net.", ""):v for k,v in state.items()}

torch.save(state, "jit.pth")