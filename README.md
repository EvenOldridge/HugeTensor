# HugeTensor
HugeTensor Bug in Pytorch/NVidia V100 Memory access

When I load giant tensors into gpu memory there comes a point where the throughput starts to degrade.
At first it's a bit faster than CPU memory, but after a fixed memory size access gets slower
If I setup the dataloader to access the later memory locations at the start then the slowdown is immediate 

It seems like there's an issue, either on the pytorch side or with nvidia memory when accessing a single large tensor.

I'm not sure if it's specific to the dgx-1 / v100 cards or if it's a general issue that just hasn't been discovered because noone's put that big of a tensor into GPU memory before. (I'm throwing 10 gigs+ on)

I've tested:
 - Different tensor shapes (2x wider tensors results in degredation at ~6M samples)
 - Changed the tensor size to be within the limit (no slowdown occurs)
 - CPU Memory (No issues)
 - Different GPUs, although all on the same DGX-1 (hopefully validating hardware is functioning)
 - Artifically starting the dataloader at the point where it slows down. (Slowdown is immediate and performance is much worse (25K samples/s)
 - **Breaking up the tensor into multiple blocks to see if being contiguous matters** (multiple tensors don't have the same effect.  Only one larger tensor demonstrates this issue)
 - Started with multiple blocks and concated together on the GPU.  (issue shows up again.  seems to indicate a possible indexing issue.
 - Tested just the indexing of the single large random tensor to see if it was impacted (no slowdown)

**This is possibly due to a int (or long) variable in the memory addressing of the tensor**
If I calculate the size of the tensor where the slowdown occurs (45x4bytesx~12M = 2.16B) that's suspiciously close to the in limit of 2147483647

What's strange is that the slowdown only occurs if there is significant access in that range.  I tested a tensor that was 4 bytes larger (and one that was 100K larger) and neither of those displayed significant problems.  It's only when it's much larger that it seems to cause the issue.
