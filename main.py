import numpy as np
import pygame
import sys

from heapq import *

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
    with the number of neighbors form the convolution and the current state of the
    cell.
    """
    # pad the array, the size of the kernel is always 3
    img_pad = np.zeros((W+2, H+2), dtype=np.uint8)
    img_res = image.copy()

    for _ in range(n):
        img_aux = img_res.copy()
        img_pad[1:W+1, 1:H+1] = img_res

        # https://stackoverflow.com/a/43087507
        sub_mats = np.lib.stride_tricks.as_strided(
            img_pad,
            tuple(np.subtract(img_pad.shape, (W,H)) + 1) + (W,H),
            img_pad.strides * 2
        )

        # number of neighbors for each cell
        neigt = np.einsum('ij,ijkl->kl', KERNEL, sub_mats)

        for i in range(W):
            for j in range(H):
                img_res[i,j] = RULE[img_aux[i, j], neigt[i, j]]

    return img_res


def make_map():
    """
    Make a random cave like map using automata rules. Return the map and
    the image of the map.
    """
    img = rand_img()
    res = convolve(img, RULE1)
    res = convolve(res, RULE2, 5)

    res[::] *= 255
    
    aux = np.zeros((15, 15), dtype=np.uint8)
    aux[0:-1, 0:-1]=1
    # expand each tile from 1px x 1px to 15px x 15px
    img = np.kron(res, aux)
    img = np.stack((img,)*3, axis=-1) # grayscale to RGB(3 channels)

    return res, img


def render(img):
    """Render img to WIN."""
    
    # highlight the current selected tile
    x, y = pygame.mouse.get_pos()
    x = int(x - x % 15)
    y = int(y - y % 15)

    img[x:x+14, y:y+14] = np.array([111, 155, 202], dtype=np.uint8)
    
    # highlight the visited tiles
    if VIST is not None:
        for x, y in VIST:
            img[x*15:x*15+14, y*15:y*15+14] = np.array([128, 128, 128], dtype=np.uint8)

    # highlight the path
    if PATH is not None:
        for x, y in PATH:
            img[x*15:x*15+14, y*15:y*15+14] = np.array([202, 103,  74], dtype=np.uint8)

    # highlight start tile
    if START is not None:
        x, y = START
        img[x*15:x*15+14, y*15:y*15+14] = np.array([150, 169, 103], dtype=np.uint8)

    # highlight end tile
    if END is not None:
        x, y = END
        img[x*15:x*15+14, y*15:y*15+14] = np.array([211, 169,  74], dtype=np.uint8)

    WIN.blit(pygame.surfarray.make_surface(img), (0, 0))
    pygame.display.flip()


def get_mpos():
    """Mouse position in MAP coordinates."""
    x, y = pygame.mouse.get_pos()
    return int(x/15), int(y/15)


def astar(start, end, m, h):
    # NOTE: m is padded so that we don't have to check for boundaries

    neighbors = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
    
    min_heap = []

    gscore = np.full((W+2, H+2), np.inf)
    fscore = np.full((W+2, H+2), np.inf)

    path = {}
    global VIST
    VIST = set()

    gscore[start[0], start[1]] = 0
    fscore[start[0], start[1]] = h(start, end)

    heappush(min_heap, (fscore[start[0], start[1]], (start[0], start[1])))

    while min_heap:

        curr = np.array(heappop(min_heap)[1])

        if (curr == end).all():
            ret_path = []

            c = (curr[0], curr[1])

            while c in path:
                ret_path.append((c[0] - 1, c[1] - 1)) # remove the padding
                c = path[c]
            return ret_path

        for n in neighbors:
            wn = curr + n

            if m[wn[0], wn[1]] == 255: # only 255 are valid neighbors
                t_gscore = gscore[curr[0], curr[1]] + 1

                VIST.add((wn[0]-1, wn[1]-1))

                if t_gscore < gscore[wn[0], wn[1]]:
                    path[(wn[0], wn[1])] = (curr[0], curr[1])
                    gscore[wn[0], wn[1]] = t_gscore
                    fscore[wn[0], wn[1]] = gscore[wn[0], wn[1]] + h(wn, end)

                    if (wn[0], wn[1]) not in [i[1] for i in min_heap]:
                        heappush(min_heap, (fscore[wn[0], wn[1]], (wn[0], wn[1])))

    return None


if __name__ == "__main__":
    pygame.init()
    pygame.display.set_caption('Path-Finding')

    START = None
    END = None
    PATH = None
    VIST = None

    WIN = pygame.display.set_mode((W*15, H*15), 0, 24)

    MAP, IMG = make_map()

    CLOCK = pygame.time.Clock()

    while True:
        CLOCK.tick(30) # set FPS to 30
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_q):
                print('')
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_r:
                START = None
                END = None

                MAP, IMG = make_map()
            elif event.type == KEYDOWN and event.key == K_a:
                if START is not None and END is not None:
                    # pad MAP
                    m = np.zeros((W+2, H+2), dtype=np.uint8)
                    m[1:-1, 1:-1] += MAP
                    # euclidean distance heuristic

                    PATH = astar(
                        np.array(START) + 1,
                        np.array(END) + 1,
                        m,
                        lambda x, y: (y[0]-x[0])**2 + (y[1]-x[1])**2
                    )
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

        render(IMG.copy())
        print(f'FPS: {int(CLOCK.get_fps())}\r', end='')
