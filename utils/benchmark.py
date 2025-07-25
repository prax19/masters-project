import torch
from torch.utils.data import DataLoader
from utils.image import *

def start_benchmark(
    model,
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    model.to(device).eval().half()

    dummy = torch.randn(1, 3, 1024, 2048, device="cuda").half()  # batch = 1, 1024×2048

    #w warm up
    torch.backends.cudnn.benchmark = True       # autotuner cuDNN
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)

    reps = 800
    starter = torch.cuda.Event(True); ender = torch.cuda.Event(True)
    timings, allocs, resvs, peaks = [], [], [], []

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for _ in range(reps):
            starter.record()
            _ = model(dummy)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))  # ms
            allocs.append(torch.cuda.memory_allocated(device))
            resvs.append(torch.cuda.memory_reserved(device))
            peaks.append(torch.cuda.max_memory_allocated(device))

    lat_ms = sum(timings) / len(timings)
    mean_alloc = sum(allocs)/len(allocs)/(1024**2)
    mean_resv  = sum(resvs)/len(resvs)/(1024**2)
    peak_alloc = max(peaks)/(1024**2)
    print(f"Średnia latencja : {lat_ms:.3f} ms  |  FPS : {1000/lat_ms:.1f} | Alloc: {mean_alloc:.1f} MiB | Reserved: {mean_resv:.1f} MiB | Peak: {peak_alloc:.1f} MiB")

    return lat_ms, 1000/lat_ms, mean_alloc, mean_alloc, peak_alloc