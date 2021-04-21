import numpy as np
import pygame
import sys

# for keys
from pygame.locals import *

W = 50
H = 50

KERNEL = np.ones((3, 3), dtype=np.uint8)
KERNEL[1,1] = 0

# B678/S345678
RULE1 = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1],  # Born
                  [0, 0, 0, 1, 1, 1, 1, 1, 1]]) # Survive

# B5678/S45678
RULE2 = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1],  # Born
                  [0, 0, 0, 0, 1, 1, 1, 1, 1]]) # Survive


def rand_img():
    """Make a random image, half white, half black."""
    img = np.ones(W * H, dtype=np.uint8)
    img[:int(W * H / 2)] = 0
    np.random.shuffle(img)
    return np.reshape(img, (W, H))


def convolve(image, RULE, n=1):
    """
    Convolve KERNEL over image, cell result(born or survive) is indexed using RULE
    with the number of neighbors form the convolution and the current stat of the
    cell.
    """
    # pad the array, the size of the kernel is always 3
    img_pad = np.zeros((W + 2, H + 2), dtype=np.uint8)
    img_res = image.copy()

    for _ in range(n):
        img_aux = img_res.copy()
        img_pad[1:W + 1, 1:H + 1] = img_res

        # https://stackoverflow.com/a/43087507
        sub_mats = np.lib.stride_tricks.as_strided(
            img_pad,
            tuple(np.subtract(img_pad.shape, (W,H)) + 1) + (W,H),
            img_pad.strides * 2
        )

        # number of neighbors for each cell
        neigt = np.einsum('ij,ijkl->kl', KERNEL, sub_mats)

        # TODO: re-write using numpy
        for i in range(W):
            for j in range(H):
                img_res[i,j] = RULE[img_aux[i, j], neigt[i, j]]

    return img_res


def make_map():
    """Make a random cave like map using automata rules."""
    img = rand_img()
    res = convolve(img, RULE1)
    res = convolve(res, RULE2, 5)

    res[::] *= 255
    return res


def render():
    """Render MAP to WIN."""

    # expand each tile from 1px x 1px to 15px x 15px
    IMG = np.kron(MAP, np.ones((15, 15), dtype=np.uint8))
    IMG = np.stack((IMG,)*3, axis=-1) # grayscale to RGB(3 channels)

    # highlight the current selected tile
    x, y = pygame.mouse.get_pos()
    x = int(x - x % 15)
    y = int(y - y % 15)

    IMG[x:x + 15, y:y + 15] = np.array([111, 155, 202], dtype=np.uint8)

    # highlight start tile
    if START is not None:
        x, y = START
        IMG[x * 15:x * 15 + 15, y * 15:y * 15 + 15] = np.array([150, 169, 103], dtype=np.uint8)

    # highlight end tile
    if END is not None:
        x, y = END
        IMG[x * 15:x * 15 + 15, y * 15:y * 15 + 15] = np.array([211, 169,  74], dtype=np.uint8)

    WIN.blit(pygame.surfarray.make_surface(IMG), (0, 0))
    pygame.display.flip()


def get_mpos():
    """Mouse position in MAP coordinates."""
    x, y = pygame.mouse.get_pos()
    return int(x / 15), int(y / 15)


if __name__ == "__main__":
    pygame.init()
    pygame.display.set_caption('Path-Finding')

    WIN = pygame.display.set_mode((W * 15, H * 15), 0, 24)

    MAP = make_map()

    START = None
    END = None

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_q):
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_r:
                START = None
                END = None

                MAP = make_map()
            elif event.type == KEYDOWN and event.key == K_a:
                pass
            elif event.type == KEYDOWN and event.key == K_d:
                pass
            elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                x, y = get_mpos()
                if not MAP[x, y] == 0:
                    START = x, y
            elif event.type == MOUSEBUTTONDOWN and event.button == 3:
                x, y = get_mpos()
                if not MAP[x, y] == 0:
                    END = x, y

        render()
