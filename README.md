# HugeTensor
HugeTensor Bug in Pytorch/NVidia V100 Memory access

When I load giant tensors into gpu memory there comes a point where the throughput starts to degrade.
At first it's a bit faster than CPU memory, but after a fixed memory size access gets slower
If I setup the dataloader to access the later memory locations at the start then the slowdown is immediate 

It seems like there's an issue, either on the pytorch side or with nvidia memory when accessing a single large tensor.

I'm not sure if it's specific to the dgx-1 / v100 cards or if it's a general issue that just hasn't been discovered because noone's put that big of a tensor into GPU memory before. (I'm throwing 10 gigs+ on)

I've tested:
 - Different tensor shapes (2x wider tensors results in degredation at ~6M samples)
 - CPU Memory (No issues)
 - Different GPUs, although all on the same DGX-1 (hopefully validating hardware is functioning)
 - Artifically starting the dataloader at the point where it slows down. (Slowdown is immediate and performance is much worse (25K samples/s)
 - **Breaking up the tensor into multiple blocks to see if being contiguous matters** (multiple tensors don't have the same effect.  Only one larger tensor demonstrates this issue)
 - Tested just the indexing of the single large tensor (no slowdown)
 
