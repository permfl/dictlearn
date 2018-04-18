### Trained Dictionaries


`100k_natural_img_patches`
Trained by Elad et al in for original ksvd paper
Trained on 100 000 random patches from different clean natural images


`DOG_20k.npy`
Trained on noisy patches from images/test/dog, 64x144


`OBAMA_16x16.npy`
100k iterations on OBAMA 60k patches

`OBAMA_20k`
20k iters on 64x144 obama


#### Plans

Will move this into the module such that its possible to import already trained dictionaries. What I want is to be able to write:

`from segment.dictionaries import some_pretrained_dict`

with `some_pretrained_dict` probably being saved in `~/.segment/some_pretrained_dict`. Not sure if it's possible to do it this way, but will look into it



