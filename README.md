This repo contains the source code and instructions to run SOLIS, a grandmaster-level chess engine
that plays entirely within a contrastive embedding space, by selecting actions that move toward
known favorable regions. A 3D orbit of the embedding space is shown below:

![til](orbit.gif)

The learned manifold is linear, and each side corresponds to one color winning. On one side, 
white has checkmate, and on the other black has checkmate. Equal positions lie near the center of
the embedding space, and you can traverse entirely with embedding arithmetic.

In our paper, we suggest a planning algorithm based on an advantage direction, though other
planning processes are possible.
