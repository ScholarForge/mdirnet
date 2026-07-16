import time
import torch
from thop import profile

from models.mdirnet import MDIRNET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MDIRNET().to(device)
model.eval()

dummy = torch.randn(1,3,256,256).to(device)


params = sum(p.numel() for p in model.parameters())


flops, params_thop = profile(
    model,
    inputs=(dummy,),
    verbose=False
)

print(f"Parameters : {params/1e6:.2f} M")
print(f"FLOPs      : {flops/1e9:.2f} GFLOPs")



if device.type == "cuda":

    # warmup
    for _ in range(20):
        _ = model(dummy)

    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    repetitions = 100
    timings = []

    with torch.no_grad():

        for _ in range(repetitions):

            starter.record()
            _ = model(dummy)
            ender.record()

            torch.cuda.synchronize()

            timings.append(starter.elapsed_time(ender))

    print(f"Average runtime : {sum(timings)/len(timings):.2f} ms")

else:

    start = time.time()

    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy)

    end = time.time()

    print(f"Average runtime : {(end-start)/50*1000:.2f} ms")