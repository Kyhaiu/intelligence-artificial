import numpy as np
import pygame
import sys

# for keys
from pygame.locals import *

W = 100
H = 100

KERNEL = np.ones((3, 3), dtype=np.uint8)
KERNEL[1,1] = 0

# B678/S345678
RULE1 = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1],  # Born
                  [0, 0, 0, 1, 1, 1, 1, 1, 1]]) # Survive

# B5678/S45678
RULE2 = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1],  # Born
                  [0, 0, 0, 0, 1, 1, 1, 1, 1]]) # Survive


# make a random image, half white, half back
def rand_img():
    img = np.ones(W * H, dtype=np.uint8)
    img[:int(W * H / 2)] = 0
    np.random.shuffle(img)
    return np.reshape(img, (W, H))


def convolve(image, RULE, n=1):
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
    img = rand_img()
    res = convolve(img, RULE1)
    res = convolve(res, RULE2, 5)

    res[::] *= 255
    return pygame.surfarray.make_surface(res)


if __name__ == "__main__":
    pygame.init()
    WIN = pygame.display.set_mode((W, H), 0, 32)

    img = make_map()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_q):
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_r:
                img = make_map()

        WIN.blit(img, (0, 0))
        pygame.display.flip()
