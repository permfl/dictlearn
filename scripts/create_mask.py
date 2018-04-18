#!/usr/bin/env python
from __future__ import print_function
import os
import argparse
import textwrap
import numpy as np
import dictlearn as dl


class ImageOpen(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(ImageOpen, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, path, option_string=None):
        if path is None:
            setattr(namespace, self.dest, (None, None))

        if not os.path.isfile(path):
            raise ValueError('Cannot find image %s' % path)

        _, ext = os.path.splitext(path)

        if ext == '.npy':
            image = np.load(path)
        else:
            image = dl.imread(path)

        setattr(namespace, self.dest, (image, path))


def is_rgb(image):
    """
        Check if an image is RGB type. True mean probably True,
        False definitely not RGB image
    """
    return (
        image.ndim == 3 and image.shape[-1] == 3
        and image.min() >= 0 and image.max() <= 255
    )


def subtract_mask(image, drawing):
    """
        Create a mask by drawing on the image drawing, then extract this
        mask by subtracting image from drawing

    :param image:
        Original image
    :param drawing:
        Original image with mask drawn on top
    :return:
        Difference of drawing and image
    """
    if is_rgb(drawing):
        drawing = dl.rgb2gray(drawing)

    if is_rgb(image):
        image = dl.rgb2gray(image)
        thresh = 3
        return np.abs(drawing - image) < thresh

    if image.ndim == 2 and drawing.ndim == 3 and drawing.shape[2] == 3:
        drawing = dl.rgb2gray(drawing)
        thresh = 0.5
    else:
        raise ValueError('Incompatible shapes, {} and {}'
                         .format(image.shape, drawing.shape))

    return np.abs(drawing - image) < thresh


def extract_color(image, color):
    assert image.dtype.kind in 'iu', 'Need image of integer type'

    if image.ndim == 3:
        if image.shape[2] in [3, 4]:
            # RGB[A] image
            if len(color) == 1:
                color.append(color[0])
                color.append(color[0])
            elif len(color) != 3:
                raise ValueError('Need RGB color for RGB image, not {}'
                                 .format(color))
            if image.shape[2] == 4:
                # RGBA - Discard alpha
                image = image[:, :, :-1]

            mask = np.logical_not(np.all(image[:, :] == color, axis=2))
        else:
            # Standard 3D image
            if len(color) != 1:
                raise ValueError('Can only take one color value for greyscale image, not'
                                 ' {}'.format(len(color)))

            mask = image[:, :, :] != color[0]
    elif image.ndim == 2:
        # 2D greyscale
        if len(color) != 1:
            raise ValueError('Cannot extract color {} from greyscale image, color '
                             'has to be a single number, grey value'.format(color))

        mask = image[:, :] != color[0]
    else:
        raise ValueError('Can only extract mask from 2D or 3D images, not {}D'
                         .format(image.ndim))

    return mask


def create_geometry(image, point):
    # todo finish
    raise NotImplementedError()


def write_mask(mask, path, output=None):
    if output is None:
        name, ext = os.path.splitext(path)
        output = '{}_mask{}'.format(name, ext)

    _, ext = os.path.splitext(output)

    if ext == '.npy':
        np.save(output, mask)
    else:
        dl.imsave(output, mask)

    print('Mask saved to: {}'.format(output))


if __name__ == '__main__':
    description = textwrap.dedent("""\
    Script for creating a boolean mask for some image
    
    Possible ways of creating the mask:
    
    Drawing:
        If '--drawing path/to/some/image' argument is used the it's 
        assumed that 'path/to/some/image' is the input image with 
        some drawing up top. The drawing image should be RGB for 
        the best results (input image need not be). The mask is 
        created by subtracting the input image from the drawing.
        
    Extract colored region:
        If '--color R G B' argument is set then every pixel in the 
        input image is marked for inpainting if its color is identical 
        to [R, G, B]. The input image should be RGB for the best 
        results. It's possible for the input image to be grayscale and 
        then extract a single color value, but the results from this 
        will not be as good
         
    Random mask:
        If '--random' is used then a random mask is created. 
        The number of pixels marked for inpainting is determined 
        by '--ratio [0.0 - 1.0]'. If ratio is set to 0.7 then 
        30%% of the pixels will be removed.
        
    Points:
        Not used
    """)

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'input', type=str, action=ImageOpen, help='Create mask for this image'
    )

    parser.add_argument(
        '-d', '--drawing', type=str, action=ImageOpen,
        help='Image with mask drawn on top. Input image '
             'is subtracted from this to create the mask'
    )

    parser.add_argument(
        '-c', '--color', nargs='+', type=int,
        help='RGB Color. Mark every thing in input image of this color as mask'
    )

    parser.add_argument(
        '-p', '--point', nargs='*', type=str,
        help='List of points, to mark the boundaries of the mask '
             'Two points: Line, three points: Triangle, four points: Box'
    )

    parser.add_argument(
        '-r', '--random', action='store_true',
        help='Create random mask for image, adjust corrution level with --ratio'
    )

    parser.add_argument(
        '--ratio', type=float, default=0.5,
        help=r'How many of the pixels to keep if random mask. 0.7 keeps 70%% of the '
             'original data, and removes 30%%'
    )

    parser.add_argument(
        '-o', '--output', type=str, help='Where to save the mask, with image format.'
    )

    args = parser.parse_args()

    inp_image, inp_path = args.input

    mask = None

    if args.drawing is not None:
        mask = subtract_mask(inp_image, args.drawing[0])
    elif args.color is not None:
        mask = extract_color(inp_image, args.color)
    elif args.point is not None:
        mask = create_geometry(inp_image, args.point)
    elif args.random:
        if not is_rgb(inp_image) and inp_image.ndim == 3:
            raise SystemExit('Cannot create random mask from 3D image')
        h, w = inp_image.shape
        mask = np.random.rand(h, w) < args.ratio
    else:
        raise SystemExit(
            'Cannot create mask. Need at least one method for generating the mask.\n'
            'Run: python {} --help'.format(parser.prog)
        )

    write_mask(mask, inp_path, args.output)
