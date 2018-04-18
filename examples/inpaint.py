import os
import numpy as np
import matplotlib.pyplot as plt
import dictlearn as dl

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

plt.rcParams['image.cmap'] = 'bone'


house = os.path.join(base_dir, 'images/test/house.png')
lena = os.path.join(base_dir, 'images/test/lena.png')
text_mask = os.path.join(base_dir, 'images/test/TextMask256.png')

house = dl.imread(house).astype(float)
lena = dl.imread(lena).astype(float)
text_mask = dl.imread(text_mask).astype(bool)

keep = 0.3  # Keep 30% of the original data
random_mask = np.random.rand(*lena.shape) < keep

# We now have two images, and two masks - we'll apply both masks to
# both images and see how the structure of the image affect the result
plt.subplot(221)
plt.imshow(house)
plt.axis('off')

plt.subplot(222)
plt.imshow(lena)
plt.axis('off')

plt.subplot(223)
plt.imshow(text_mask)
plt.axis('off')

plt.subplot(224)
plt.imshow(random_mask)
plt.axis('off')
plt.figure()

# Corrupt the images
house_text = house*text_mask
house_rnd = house*random_mask
lena_text = lena*text_mask
lena_rnd = lena*random_mask

plt.suptitle('Corrupted images')
plt.subplot(221)
plt.imshow(house_text)
plt.axis('off')

plt.subplot(222)
plt.imshow(lena_text)
plt.axis('off')

plt.subplot(223)
plt.imshow(house_rnd)
plt.axis('off')

plt.subplot(224)
plt.imshow(lena_rnd)
plt.axis('off')
plt.figure()

iters = 10


def create_callback(original_image):
    def print_iter(image_estimate, iteration):
        psnr = dl.utils.psnr(original_image, image_estimate, 255)
        print('Iter %d, PSNR = %.2f' % (iteration + 1, psnr))

    return print_iter


inpaint = dl.Inpaint(house_text, text_mask)
house_text_inpainted = inpaint.inpaint(callback=create_callback(house))

inpaint = dl.Inpaint(lena_text, text_mask)
lena_text_inpainted = inpaint.inpaint(callback=create_callback(lena))

inpaint = dl.Inpaint(house_rnd, random_mask)
house_rnd_inpainted = inpaint.inpaint(callback=create_callback(house))

inpaint = dl.Inpaint(lena_rnd, random_mask)
lena_rnd_inpainted = inpaint.inpaint(callback=create_callback(lena))

plt.suptitle('Each of these are the cleaned version of img in same spot as prev plot')
plt.subplot(221)
plt.imshow(house_text_inpainted)
plt.title('PSNR = {:.3f}'.format(dl.utils.psnr(house, house_text_inpainted, 255)))
plt.axis('off')

plt.subplot(222)
plt.imshow(lena_text_inpainted)
plt.title('PSNR = {:.3f}'.format(dl.utils.psnr(lena, lena_text_inpainted, 255)))
plt.axis('off')

plt.subplot(223)
plt.imshow(house_rnd_inpainted)
plt.title('PSNR = {:.3f}'.format(dl.utils.psnr(house, house_rnd_inpainted, 255)))
plt.axis('off')

plt.subplot(224)
plt.imshow(lena_rnd_inpainted)
plt.title('PSNR = {:.3f}'.format(dl.utils.psnr(lena, lena_rnd_inpainted, 255)))
plt.axis('off')
plt.show()
