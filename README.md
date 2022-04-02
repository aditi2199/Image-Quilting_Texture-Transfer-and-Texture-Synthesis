# ImageQuilting_TextureTransferTextureSynthesis
Image Quilting Algorithm*:
○ Go through the image to be synthesized in raster scan order in steps of one block (minus the 
overlap).
○ For every location, search the input texture for a set of blocks that satisfy the overlap 
constraints (above and left) within some error tolerance. Randomly pick one such block. 
○ Compute the error surface between the newly chosen block and the old blocks at the overlap 
region. Find the minimum cost path along this surface and make that the boundary of the new 
block. Paste the block onto the texture. Repeat

Minimum Error Boundary Cut
Ei,j = ei,j + min(Ei-1,j-1, Ei-1,j, Ei-1,j+1)
● After DP above, in the end, the minimum value of the last row in E will indicate 
the end of the minimal vertical path though the surface and we can trace back 
and find the path of the best cut through the overlapped region.


Texture Transfer
● If we modify the synthesis algorithm by requiring that each patch satisfy a 
desired correspondence map, C , as well as satisfy the texture synthesis 
requirements, we can use it for texture transfer.
● The correspondence map is a spatial map of some corresponding quantity 
over both the texture source image and a controlling target image.
● For texture transfer, image being synthesized must respect two independent 
constraints:
○ (a) the output are legitimate, synthesized examples of the source texture
○ (b) that the correspondence image mapping is respected.
● Hence, we modify the error term by the use of an ‘alpha’ parameter.
