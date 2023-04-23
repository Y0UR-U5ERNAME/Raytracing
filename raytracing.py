import sys
from rtools import ray, cam2world0, cam2world, normalize, trace, vector, randcirclept, RAYSPERPIXEL, colorclamp, pi, sqrt
from random import seed, shuffle
from itertools import product
from numpy import array, zeros

import cProfile

def frag(i, j):
    global prodi, frame, pxa, rpxa

    sumlight = vector((0, 0, 0))
    for x in range(RAYSPERPIXEL):
        r = ray()
        djitter = randcirclept(0) # defocus jitter
        r.pos = cam2world0(*djitter)

        jitter = randcirclept(px / sqrt(pi)) # regular jitter
        viewpoint = cam2world(i - 250 + jitter[0], 250 - j + jitter[1])
        r.dir = normalize(viewpoint - r.pos)

        c = trace(r)
        del r
        sumlight = sumlight + c

    c = sumlight / RAYSPERPIXEL
    rpxa[i//px, j//px] = rpxa[i//px, j//px]*(1-weight) + c*weight
    out = colorclamp(rpxa[i//px, j//px])
    pxa[i:i+px, j:j+px] = out

def main():
    global prodi, frame, pxa, weight

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    i, j = p[prodi]
    seed(i + j * 500//px + frame * 500*500//(px*px))
    frag(i, j)
    prodi += 1
    if prodi == len(p):
        prodi = 0
        frame += 1
        weight = 1 / (frame + 1)
        print(frame)
    
    if 1 or prodi % (len(p) // 25) == 0: pygame.display.flip()

if __name__ == '__main__':
    from os import environ
    environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    import pygame

    pygame.init()
    Screen = pygame.display.set_mode((500,500))
    Screen.fill("Black")
    pxa = pygame.PixelArray(Screen)
    px = 10 # pixelization
    rpxa = zeros((500//px,500//px,3))

    p = list(product(range(0,500,px), repeat=2))
    shuffle(p)
    p = tuple(p)
    prodi = 0
    frame = 0
    weight = 1
    while 1: main()
    #cProfile.run('for I in range(20*len(p)): main()')