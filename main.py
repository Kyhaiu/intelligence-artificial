import numpy as np
import math
import pygame
import sys
import time

from heapq import *
# for keys
from pygame.locals import *
from queue import *

"""
    Código criado por:
        - Lucas Fischer Mulling;
        - Marcos Augusto Campagnaro Mucelini

    PROBLEMA ESCOLHIDO: Pathfind
    ALGORITMO ESCOLHIDOS:
        * BUSCA EM LARGURA
        * A*
"""



W = 75
H = 75

T = 10

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

    # the matrix resulting of convolution is multiply by 255, because the convolve return a matrix binary.
    # 255 is the representation of white in 8-bits.
    res[::] *= 255

    aux = np.zeros((T, T), dtype=np.uint8)
    aux[0:-1, 0:-1]=1
    # expand each tile from 1px x 1px to Tpx x Tpx
    img = np.kron(res, aux)
    img = np.stack((img,)*3, axis=-1) # grayscale to RGB(3 channels)

    return res, img


def render(img):
    """Render img to WIN."""

    # highlight the visited tiles
    if VIST is not None:
        for x, y in VIST:
            img[x*T:x*T+T-1, y*T:y*T+T-1] = np.array([128, 128, 128], dtype=np.uint8)

    # highlight the path
    if PATH is not None:
        for x, y in PATH:
            img[x*T:x*T+T-1, y*T:y*T+T-1] = np.array([202, 103,  74], dtype=np.uint8)

    # highlight start tile
    if START is not None:
        x, y = START
        img[x*T:x*T+T-1, y*T:y*T+T-1] = np.array([150, 169, 103], dtype=np.uint8)

    # highlight end tile
    if END is not None:
        x, y = END
        img[x*T:x*T+T-1, y*T:y*T+T-1] = np.array([211, 169,  74], dtype=np.uint8)

    # highlight the current selected tile
    x, y = pygame.mouse.get_pos()
    x = int(x - x % T)
    y = int(y - y % T)

    img[x:x+T-1, y:y+T-1] = np.array([111, 155, 202], dtype=np.uint8)

    WIN.blit(pygame.surfarray.make_surface(img), (0, 0))
    pygame.display.flip()


def get_mpos():
    """Mouse position in MAP coordinates."""
    x, y = pygame.mouse.get_pos()
    return int(x/T), int(y/T)


def astar(start, end, m, h):
    """A*"""
    # NOTE: m is padded so that we don't have to check for boundaries

    s = time.time()

    neighbors = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
    # neighbors order
    #   +-+
    #   |1|
    # +-+-+-+
    # |4| |2|
    # +-+-+-+
    #   |3|
    #   +-+

    min_heap = [] # conjunto aberto de elementos(binary heap)

    # inicializa os mapas g e f com inf (representação maxima de inteiros da lib numpy).
    gscore = np.full((W+2, H+2), np.inf)
    fscore = np.full((W+2, H+2), np.inf)

    path = {}
    vist = set()

    srt = (start[0], start[1])

    gscore[srt] = 0
    fscore[srt] = h(start, end)

    heappush(min_heap, (fscore[srt], srt))

    while min_heap:

        curr = np.array(heappop(min_heap)[1])

        if (curr == end).all():
            # chegamos no destino, reconstruir o caminho
            e = round(time.time() - s, 4)

            ret_path = []

            c = (curr[0], curr[1])

            while c in path:
                ret_path.append((c[0]-1, c[1]-1)) # remove the padding
                c = path[c]

            print(f':: A*\n dist: {len(ret_path)}\n vist: {len(vist)}\n time: {e}')

            return ret_path, vist

        for n in neighbors:
            wn = (curr[0] + n[0], curr[1] + n[1])

            if m[wn] == 255: # apenas vizinhos com valor 255 são validos

                t_gscore = gscore[curr[0], curr[1]] + 1 # a distancia entre cada vizinho sempre eh 1

                vist.add((wn[0]-1, wn[1]-1)) # marcar o vizinho como visitado (removendo o padding)

                if t_gscore < gscore[wn]:
                    # achamos um caminho com uma distancia melhor distancia
                    path[wn] = (curr[0], curr[1])
                    gscore[wn] = t_gscore
                    fscore[wn] = gscore[wn] + h(wn, end) # heuristica usada aqui
                    heappush(min_heap, (fscore[wn], wn))

                    if wn not in [i[1] for i in min_heap]:
                        # se o vizinho não esta no monte(heap) adicioná-lo
                        heappush(min_heap, (fscore[wn], wn))

    return None, vist


def bfs(start, end, m):
    """Breadth-first search."""
    # NOTE: m is padded

    s = time.time()

    queue = Queue()

    queue.put((start[0], start[1]))

    vist = set()
    vist.add((start[0], start[1]))

    path = {}

    neighbors = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
    # neighbors order
    #   +-+
    #   |1|
    # +-+-+-+
    # |4| |2|
    # +-+-+-+
    #   |3|
    #   +-+

    while not queue.empty():

        curr = queue.get()

        if curr[0] == end[0] and curr[1] == end[1]:
            # chegamos no destino, reconstruir o caminho
            e = round(time.time() - s, 4)

            ret_path = []

            c = curr

            while c in path:
                ret_path.append((c[0]-1, c[1]-1)) # remove o padding
                c = path[c]

            print(f':: BFS\n dist: {len(ret_path)}\n vist: {len(vist)}\n time: {e}')

            return ret_path, [(i[0]-1, i[1]-1) for i in vist]

        for n in neighbors:
            wn = (curr[0] + n[0], curr[1] + n[1])

            if m[wn[0], wn[1]] == 255:
                if wn not in vist:
                    vist.add(wn)
                    path[wn] = curr
                    queue.put(wn)

    return None, vist


if __name__ == "__main__":
    pygame.init()
    pygame.display.set_caption('Path-Finding')

    START = None
    END = None
    PATH = None
    VIST = None

    WIN = pygame.display.set_mode((W*T, H*T), 0, 24)

    MAP, IMG = make_map()

    # CLOCK = pygame.time.Clock()

    if len(sys.argv) == 2 and sys.argv[1] == 'm':
        # manhattan distance heuristic
        heuristic = lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1])
    else:
        # sum of squares heuristic
        heuristic = lambda x, y: (y[0]-x[0])**2 + (y[1]-x[1])**2

    while True:
        # CLOCK.tick(60) # set FPS to 30
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_q):
                # print('')
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN and event.key == K_r:
                START = None
                END = None
                VIST = None
                PATH = None

                MAP, IMG = make_map()
            elif event.type == KEYDOWN and event.key == K_a:
                if START is not None and END is not None:
                    # pad MAP
                    padded_map = np.zeros((W+2, H+2), dtype=np.uint8)
                    padded_map[1:-1, 1:-1] += MAP

                    PATH, VIST = astar(
                        np.array(START) + 1,
                        np.array(END) + 1,
                        padded_map,
                        heuristic
                    )
            elif event.type == KEYDOWN and event.key == K_d:
                if START is not None and END is not None:
                    # pad MAP
                    padded_map = np.zeros((W+2, H+2), dtype=np.uint8)
                    padded_map[1:-1, 1:-1] += MAP

                    PATH, VIST = bfs((START[0] + 1, START[1] + 1), (END[0] + 1, END[1] + 1), padded_map)
            elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                x, y = get_mpos()
                if not MAP[x, y] == 0:
                    START = x, y
            elif event.type == MOUSEBUTTONDOWN and event.button == 3:
                x, y = get_mpos()
                if not MAP[x, y] == 0:
                    END = x, y

        render(IMG.copy())
        # print(f'FPS: {int(CLOCK.get_fps())}\r', end='')
