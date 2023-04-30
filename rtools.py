from math import dist, sin, cos, tan, sqrt, pi
from random import gauss, random
from numpy import dot, array, cross, maximum, minimum

FOV = 90 * pi/180
MAXBOUNCES = 10
FLEN = tan(FOV/2)*500
PLANEZ = 10-4
RAYSPERPIXEL = 1

vector = array

class camera():
    def __init__(self):
        self.pos = vector((0, 3, -10))
        self.dir = vector((-0.3, 0, 0))
cam = camera()

class ray():
    def __init__(self):
        self.pos = vector((0, 0, 0))
        self.dir = vector((0, 0, 0))

class sphere():
    def __init__(self, center, radius, mtl):
        self.center = center
        self.radius = radius
        self.mtl = mtl

class triangle():
    def __init__(self, a, b, c, mtl, na=None, nb=None, nc=None):
        self.a, self.b, self.c = a, b, c
        if na == None: na = cross(b - a, c - a); nb = na; nc = na
        self.na, self.nb, self.nc = na, nb, nc
        self.mtl = mtl

class material():
    def __init__(self, color, emstrength=0, emcolor=vector((255,255,255)), smoothness=0, specprob=0, speccolor=vector((255,255,255))):
        self.color = (color / 255)**2.2
        self.emstrength = emstrength
        self.emcolor = (emcolor / 255)**2.2
        self.smoothness = smoothness
        self.specprob = specprob
        self.speccolor = (speccolor / 255)**2.2

def cam2world(px, py):
    camrel = prot(px/FLEN*PLANEZ, py/FLEN*PLANEZ, PLANEZ, *cam.dir)
    return cam.pos + camrel
def cam2world0(px, py):
    camrel = prot(px, py, 0, *cam.dir)
    return cam.pos + camrel

magnitude = lambda vec: dist(vec, (0, 0, 0))

normalize = lambda vec: (vec if not magnitude(vec) else vec / magnitude(vec))

def prot(x, y, z, ax, ay, az):
    ax, ay, az = az, ay, ax
    sa, sb, sc = sin(ax), sin(ay), sin(az)
    ca, cb, cc = cos(ax), cos(ay), cos(az)
    return vector(dot([x,y,z], 
        [
            [ca*cb, ca*sb*sc-sa*cc, ca*sb*cc+sa*sc],
            [sa*cb, sa*sb*sc+ca*cc, sa*sb*cc-ca*sc],
            [-sb, cb*sc, cb*cc]
        ]
    ))

def hitsphere(r, center, radius):
    # dst = -pos.dir - sqrt((pos.dir)^2 - (pos.pos-r^2))
    # = ( -2(O.D) - sqrt(4(O.D)^2 - 4(D.D)(O.O-r^2)) )
    #   / (2(D.D))
    offsetraypos = r.pos - center

    b = dot(offsetraypos, r.dir)
    c = dot(offsetraypos, offsetraypos) - radius*radius

    disc = b*b - c
    if disc >= 0:
        dst = (-b - sqrt(disc))
        if dst >= 0:
            hitpos = r.pos + r.dir * dst
            return (
                True,
                dst,
                hitpos,
                (hitpos - center) / radius#normalize(hitpos - center)
            )
    return (False, 0, 0, 0)

def hittriangle(r, tri):
    a, b, c, na, nb, nc = tri.a, tri.b, tri.c, tri.na, tri.nb, tri.nc

    e1 = b - a
    e2 = c - a
    n = na#cross(e1, e2)
    ao = r.pos - a
    dao = cross(ao, r.dir)

    det = -dot(r.dir, n)
    invdet = 1 / det

    u = dot(e2, dao) * invdet
    v = -dot(e1, dao) * invdet
    w = 1 - u - v
    t = dot(ao, n) * invdet
    return (
        (det >= 1e-6 and t >= 0 and u >= 0 and v >= 0 and w >= 0),
        t,
        r.pos + r.dir * t,
        normalize(na * w + nb * u + nc * v)
    )

# https://gamedev.stackexchange.com/a/18459
def hitaabb(r, lb, rt):
    df = 1 / vector(tuple((i if i else 1e-6) for i in r.dir))

    tlb = (lb - r.pos) * df
    trt = (rt - r.pos) * df

    tmin = max(minimum(tlb, trt))
    tmax = min(maximum(tlb, trt))

    return tmax >= 0 and tmin <= tmax

def hitbox(r, lb, rt, n):
    df = 1 / vector(tuple((i if i else 1e-6) for i in r.dir))

    tlb = (lb - r.pos) * df
    trt = (rt - r.pos) * df

    tmin = max(minimum(tlb, trt))
    tmax = min(maximum(tlb, trt))

    if tmax > 0 and tmin <= tmax and dot(r.dir, n) < 0:
        return (True, tmin, r.pos+r.dir*tmin, n)
    return (False,0,0,0)

def bbt(tri): # get bounding box of triangle
    a, b, c = tri.a, tri.b, tri.c
    mn = minimum(minimum(a,b),c)
    mx = maximum(maximum(a,b),c)
    return (mn, mx)

def randdir():
    return normalize(
        vector(tuple(gauss() for i in range(3)))
    )

def randcirclept(radius):
    dir = random() * 2 * pi
    dst = sqrt(random())*radius
    return (cos(dir)*dst, sin(dir)*dst)

def hitshapes(r):
    closestdist = float('inf')
    hit = False
    shapehit = None
    for s in shapes:
        if isinstance(s, sphere):
            #if not hitaabb(r, s.center-s.radius, s.center+s.radius): continue
            h = hitsphere(r, s.center, s.radius)
        else:
            #if not hitaabb(r, *bbt(s)): continue
            #h = hittriangle(r, s)
            h = hitbox(r, *bbt(s), normalize(s.na))
        if h[0] and h[1] < closestdist: hit = h; closestdist = h[1]; shapehit = s
    return hit, shapehit

def trace(r):
    color = vector((1,1,1))
    light = vector((0,0,0))
    for b in range(MAXBOUNCES + 1):
        h = hitshapes(r)
        hit, sphit = h
        if sphit:
            r.pos = hit[2]

            m = sphit.mtl

            diffdir = normalize(hit[3] + randdir())
            specdir = r.dir - hit[3] * dot(r.dir, hit[3]) * 2
            isspecbounce = m.specprob >= random()
            r.dir = lerp(diffdir, specdir, m.smoothness * isspecbounce)

            light = light + m.emcolor * m.emstrength * color
            color = color * lerp(m.color, m.speccolor, isspecbounce)
        else:
            light = light + envlight(r) * color
            break

    return light**(1/2.2) * 255

def envlight(r):
    #return vector((0,0,0))
    gradt = smoothstep(0, 0.4, r.dir[1])**0.35
    skygrad = (lerp(vector((250, 250, 250)), vector((176, 203, 245)), gradt)/255)**2.2
    sun = max(dot(r.dir, normalize(vector((-1, 1, -1)))), 0)**10 * 1

    gtst = smoothstep(-0.01, 0, r.dir[1])
    sunmask = gtst >= 1
    
    return lerp((vector((100,100,100))/255)**2.2, skygrad, gtst) + sun * sunmask

def lerp(start, end, t):
    return start*(1-t) + end*t

def smoothstep(start, end, t):
    t = min(max((t-start)/(end-start),0),1)
    return t*t*(3-2*t)

def colorclamp(color):
    if not any(color > 255): return tuple(round(i) for i in color)
    return tuple(min(round(i), 255) for i in color)

#'''
shapes = (
    sphere(vector((-2, 1, -6)), 1, material(vector((255, 255, 255)))),
    sphere(vector((0, 1, -4)), 1, material(vector((0, 100, 140)), smoothness=0.5, specprob=0.5)),
    sphere(vector((2, 1, -6)), 1, material(vector((128, 128, 128)))),

    sphere(vector((-2, 1, -2)), 1, material(vector((200, 40, 40)))),
    sphere(vector((2, 1, -2)), 1, material(vector((255, 255, 255)), smoothness=1, specprob=1)),
    #sphere(vector((1, -1, -5)), 1, material(vector((200, 200, 40)), emstrength=0.75, emcolor=vector((200, 200, 40)))),
    #sphere(vector((0, -8, 0)), 8, material(vector((200, 200, 0)))),

    triangle(vector((-8, 0, -8)), vector((-8, 0, 8)), vector((8, 0, -8)), material(vector((255, 100, 255)))),
    #triangle(vector((8, 0, -8)), vector((-8, 0, 8)), vector((8, 0, 8)), material(vector((100, 255, 100)))),
)
#'''

'''
shapes = (
    #floor
    triangle(vector((-4, -4, -4)), vector((-4, -4, 6)), vector((4, -4, -4)), material(vector((0, 128, 255)))),
    #triangle(vector((4, -4, -4)), vector((-4, -4, 6)), vector((4, -4, 6)), material(vector((0, 128, 255)))),

    #ceiling
    triangle(vector((4, 4, -4)), vector((-4, 4, 6)), vector((-4, 4, -4)), material(vector((255, 255, 255)))),
    #triangle(vector((4, 4, 6)), vector((-4, 4, 6)), vector((4, 4, -4)), material(vector((255, 255, 255)))),
    sphere(vector((0, 4, 1)), 1, material(vector((255, 255, 255)), emstrength=100)),

    #left
    triangle(vector((-4, 4, -4)), vector((-4, -4, 6)), vector((-4, -4, -4)), material(vector((255, 0, 128)))),
    #triangle(vector((-4, 4, 6)), vector((-4, -4, 6)), vector((-4, 4, -4)), material(vector((255, 0, 128)))),

    #right
    triangle(vector((4, -4, -4)), vector((4, -4, 6)), vector((4, 4, -4)), material(vector((128, 255, 0)))),
    #triangle(vector((4, 4, -4)), vector((4, -4, 6)), vector((4, 4, 6)), material(vector((128, 255, 0)))),

    #back
    triangle(vector((4, -4, 6)), vector((-4, -4, 6)), vector((-4, 4, 6)), material(vector((255, 255, 255)))),
    #triangle(vector((-4, 4, 6)), vector((4, 4, 6)), vector((4, -4, 6)), material(vector((255, 255, 255)))),

    #front
    #triangle(vector((-4, 4, -4)), vector((-4, -4, -4)), vector((4, -4, -4)), material(vector((255, 255, 255)))),
    #triangle(vector((4, -4, -4)), vector((4, 4, -4)), vector((-4, 4, -4)), material(vector((255, 255, 255)))),

    #interior
    sphere(vector((-2, -2, -1)), 2, material(vector((255, 128, 0)))),
    sphere(vector((2, -2, 3)), 2, material(vector((255, 255, 255)), smoothness=1, specprob=1)),
)
'''