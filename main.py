import numpy as np
import pygame
import math
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

    # highlight the current selected tile
    x, y = pygame.mouse.get_pos()
    x = int(x - x % 15)
    y = int(y - y % 15)

    img[x:x+14, y:y+14] = np.array([111, 155, 202], dtype=np.uint8)

    WIN.blit(pygame.surfarray.make_surface(img), (0, 0))
    pygame.display.flip()


def get_mpos():
    """Mouse position in MAP coordinates."""
    x, y = pygame.mouse.get_pos()
    return int(x/15), int(y/15)


def astar(start, end, m, h):
    """A*"""
    # NOTE: m is padded so that we don't have to check for boundaries

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

    # inicializa os mapas g e f com inf
    gscore = np.full((W+2, H+2), np.inf)
    fscore = np.full((W+2, H+2), np.inf)

    path = {}
    vist = set()

    gscore[start[0], start[1]] = 0
    fscore[start[0], start[1]] = h(start, end)

    heappush(min_heap, (fscore[start[0], start[1]], (start[0], start[1])))

    while min_heap:

        curr = np.array(heappop(min_heap)[1])

        if (curr == end).all():
            # chegamos no destino, reconstruir o caminho
            ret_path = []

            c = (curr[0], curr[1])

            while c in path:
                ret_path.append((c[0] - 1, c[1] - 1)) # remove the padding
                c = path[c]

            return ret_path, vist

        for n in neighbors:
            wn = curr + n

            if m[wn[0], wn[1]] == 255: # only 255 are valid neighbors
                                       # apenas vizinhos com valor 255 são validos

                t_gscore = gscore[curr[0], curr[1]] + 1 # a distancia entre cada vizinho eh sempre 1

                vist.add((wn[0]-1, wn[1]-1)) # add working neighbor to visited set(removing padding)
                                             # marcar o vizinho como visitado


                if t_gscore < gscore[wn[0], wn[1]]:
                    # achamos um caminho com uma distancia melhor distancia
                    path[(wn[0], wn[1])] = (curr[0], curr[1])
                    gscore[wn[0], wn[1]] = t_gscore
                    fscore[wn[0], wn[1]] = gscore[wn[0], wn[1]] + h(wn, end) # heuristica usada aqui

                    if (wn[0], wn[1]) not in [i[1] for i in min_heap]:
                        # se o vizinho não esta no monte(heap) adicioná-lo
                        heappush(min_heap, (fscore[wn[0], wn[1]], (wn[0], wn[1])))

    return None, None

def bread_first_serach(start, end):
    queue = []
    queue.append(start)

    FILLED_MAP = MAP.copy()

    while queue != []:
        x, y = queue[0][0], queue[0][1]
        queue.pop(0)

        if x == end[0] and y == end[1]:
            del queue
            break

        if x+1 < 50 and FILLED_MAP[x+1][y] == 255:
            queue.append([x+1, y])
            FILLED_MAP[x+1][y] = 100

        if y+1 < 50 and MAP[x][y+1] == 255:
            queue.append([x, y+1])
            FILLED_MAP[x][y+1] = 100

        if x-1 > 0 and FILLED_MAP[x-1][y] == 255:
            queue.append([x-1, y])
            FILLED_MAP[x-1][y] = 100

        if y-1 > 0 and FILLED_MAP[x][y-1] == 255:
            queue.append([x, y-1])
            FILLED_MAP[x][y-1] = 100


def min_distance(dist, spt):
    min_ = sys.maxsize

    for v in range(V):
        if dist[v] < min_ and spt[v] == False:
            min_ = dist[v]
            min_index = v

    return min_index

def dijkstra(self, start):
    dist = [sys.maxsize] * V
    dist[start] = 0
    short_past_tree = [False] * V

    for cout in range(V):
        u = min_distance(dist, short_past_tree)

        short_past_tree[u] = True

    for v in range(V):
        if GRAPH[u][v] > 0 and short_past_tree[v] == False and dist[v] > dist[u] + GRAPH[u][v]:
            dist[v] = dist[u] + GRAPH[u][v]

    #CHAMAR FUNCAO QUE PINTA O CAMINHO DO DIJKSTRA
    #achar um jeito de converter o mapa numa matriz de adjacencia
    #se conseguir terminar o dijktra da pra apagar a outra função que criei



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

                    PATH, VIST = astar(
                        np.array(START) + 1,
                        np.array(END) + 1,
                        m,
                        # lambda x, y: np.sum((np.abs(x - y)))       # manhattan distance heuristic
                        lambda x, y: (y[0]-x[0])**2 + (y[1]-x[1])**2 # euclidean distance heuristic
                    )
            elif event.type == KEYDOWN and event.key == K_d:
                if START is not None and END is not None:
                    bread_first_serach(START, END)
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
