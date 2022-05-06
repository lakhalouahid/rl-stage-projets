import sys
import pygame

import numpy as np

w, h, n = 360, 360, 6
a, b = w//n, h//n
aa, bb = a-8, b-8


bg_color = (0x40, 0x40, 0x40)

circ_center = (0, 0)
circ_radius = min(aa//2, bb//2)

rect_center = (0, 0)
rect_a, rect_b = aa, bb

poly_xy = [
  (0, a//2),
  (-b//2, 0),
  (0, -a//2),
  (+b//2, 0)
]

colors = [
  (200, 00, 00),
  (00, 200, 00),
  (00, 00, 200),
  (200, 200, 0),
  (0, 200, 200),
  (200, 0, 200),
]


grid = np.zeros((n, n), dtype=np.uint8)


def compute_center(idx_a, idx_b):
  return (idx_a * a + a//2, idx_b * b + b//2)

def abs_coords_pt(c: tuple, pt: tuple):
  return (c[0] + pt[0], c[1] + pt[1])

def abs_coords_pts(c: tuple, pts: list):
  return [abs_coords_pt(c, pt) for pt in pts]


pygame.init()
screen = pygame.display.set_mode(size=(w, h))
screen.fill(bg_color)

n_default = 10
color_default = 0
n_samples = 1000

for _ in range(n_default):
  color = colors[np.random.randint(len(colors))]
  idxs_a, idxs_b = np.where(grid == 0)
  if len(idxs_a) > 0:
    idx = np.random.randint(len(idxs_a))
    idx_a = idxs_a[idx]
    idx_b = idxs_b[idx]
    grid[idx_a, idx_b] = 1
    center = compute_center(idx_a, idx_b)
    object_shape = np.random.randint(3)
    if object_shape == 0:
      pygame.draw.polygon(screen, color, abs_coords_pts(center, poly_xy))
    elif object_shape == 1:
      rect = pygame.rect.Rect((center[0] - rect_a//2, center[1] - rect_b//2), (rect_a, rect_b))
      pygame.draw.rect(screen, color, rect)
    else:
      pygame.draw.circle(screen, color, center, circ_radius)


image_idx = 0
for _ in range(n_samples):
  color = colors[color_default]
  idxs_a, idxs_b = np.where(grid == 0)
  if len(idxs_a) > 0:
    idx = np.random.randint(len(idxs_a))
    idx_a = idxs_a[idx]
    idx_b = idxs_b[idx]
    grid[idx_a, idx_b] = 1
    center = compute_center(idx_a, idx_b)
    object_shape = np.random.randint(3)
    pygame.draw.circle(screen, color, center, circ_radius)
    pygame.image.save(screen, f'dataset/image-{image_idx:02}.png')
    pygame.display.flip()
    image_idx += 1
    rect = pygame.rect.Rect((center[0] - a//2, center[1] - a//2), (a, b))
    pygame.draw.rect(screen, bg_color, rect)
    pygame.time.wait(30)

