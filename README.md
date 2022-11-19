# ILGPU-CFRPlus-Subgame
A CFR+ river subgame solver for poker in .NET using ILGPU.

And my second time doing anything CUDA/GPU.

For convenience I used HoldemHand but any method of hand-ranking can be used:

https://www.codeproject.com/Articles/12279/Fast-Texas-Holdem-Hand-Evaluation-and-Analysis

Todo: 
1) Speed improvements.  Possibly solve multiple hands at a time, both players with one tree traversal, or rework the kernels to be more efficient/useful.
2) Conversion into C++/CUDA.
