import os
import matplotlib.pyplot as plt
import dictlearn as dl

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


plt.rcParams['image.cmap'] = 'bone'

image1 = os.path.join(base_dir, 'images/lena_noisy512.png')
image2 = os.path.join(base_dir, 'images/lena512.png')
noisy_image = dl.imread(image1).astype(float)
clean_image = dl.imread(image2).astype(float)


denoise = dl.Denoise(noisy_image, patch_size=11, method='online')
# method='batch' for ksvd

denoise.train(iters=1000, n_nonzero=10, n_atoms=256, n_threads=2, verbose=True)
denoised_odl = denoise.denoise(sigma=33, n_threads=2)


denosied_ksvd = dl.ksvd_denoise(noisy_image, patch_size=11, n_atoms=256, sigma=33,
                                verbose=True, n_threads=4)


plt.subplot(221)
plt.imshow(clean_image)
plt.title('Original')
plt.axis('off')

plt.subplot(222)
plt.imshow(noisy_image)
plt.title('Noisy, psnr = {:2f}'.format(dl.utils.psnr(clean_image, noisy_image, 255)))
plt.axis('off')

plt.subplot(223)
plt.imshow(denoised_odl)
plt.title('ODL, psnr = {:.2f}'.format(dl.utils.psnr(clean_image, denoised_odl, 255)))
plt.axis('off')

plt.subplot(224)
plt.imshow(denosied_ksvd)
plt.title('K-SVD, psnr = {:.2f}'.format(dl.utils.psnr(clean_image, denosied_ksvd, 255)))
plt.axis('off')
plt.show()
