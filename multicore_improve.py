import numpy as np
from scipy.spatial import KDTree
import concurrent.futures
import pygame as pg
import random

class color:
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)

class config:
    MonitorSize = (800, 600)
    FrameRate = 120
    MaxVel = 3
    MinVel = 1
    Sight = 30
    BirdNumber = 150
    flock_ratio = 1/2500
    InitVel = 3
    alignmentFactor = 0.4
    cohesionFactor = 0.2
    separationFactor = 1.5

class Game:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode(config.MonitorSize, pg.RESIZABLE)
        self.running = True
        self.clock = pg.time.Clock()
        self.fps = config.FrameRate
        self.font = pg.font.Font(None, 24)

    def draw_status(self, birds):
        fps = self.clock.get_fps()
        bird_count = len(birds.flock)
        status = f"FPS: {fps:.2f} | Bird Count: {bird_count}"
        text_surface = self.font.render(status, True, color.black)
        self.screen.blit(text_surface, (10, 10))

    def run(self, birds):
        temp = config.MonitorSize[0] * config.MonitorSize[1]
        while self.running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                elif event.type == pg.VIDEORESIZE:
                    config.MonitorSize = (event.w, event.h)
                    new_size = event.w * event.h
                    if new_size > temp:
                        birds.add(int((new_size - temp) * config.flock_ratio))
                    else:
                        birds.delete(int((temp - new_size) * config.flock_ratio))
                    temp = new_size

            self.screen.fill(color.white)
            birds.update()
            birds.draw(self.screen)
            self.draw_status(birds)
            pg.display.flip()
            self.clock.tick(self.fps)

        pg.quit()

class Bird:
    def __init__(self, pattern):
        self.pattern = pattern
        self.p = pg.Vector2(random.uniform(0, config.MonitorSize[0]), random.uniform(0, config.MonitorSize[1]))
        self.v = pg.Vector2(random.uniform(-config.InitVel, config.InitVel), random.uniform(-config.InitVel, config.InitVel))
        self.f = pg.Vector2(0, 0)

    def get_distance(self, bird):
        return bird.p - self.p

    def generate_force(self, positions, tree):
        self.f.x = 0
        self.f.y = 0

        neighbors = tree.query_ball_point([self.p.x, self.p.y], config.Sight)

        alignment_total = np.array([0.0, 0.0])
        cohesion_total = np.array([0.0, 0.0])
        separation_total = np.array([0.0, 0.0])
        total = 0

        for i in neighbors:
            bird = self.pattern.flock[i]
            if bird == self:
                continue

            vector = self.get_distance(bird)
            distance = vector.length()

            if distance < config.Sight:
                total += 1

                alignment_total += np.array([bird.v.x, bird.v.y])
                cohesion_total += np.array([vector.x, vector.y])
                separation_total += -np.array([vector.x, vector.y]) / (distance ** 2) * config.Sight

        if total != 0:
            alignment_total /= total
            cohesion_total /= total

            self.f.x += alignment_total[0] * config.alignmentFactor
            self.f.y += alignment_total[1] * config.alignmentFactor
            self.f.x += cohesion_total[0] * config.cohesionFactor
            self.f.y += cohesion_total[1] * config.cohesionFactor
            self.f.x += separation_total[0] * config.separationFactor
            self.f.y += separation_total[1] * config.separationFactor

    def update(self, positions, tree):
        self.generate_force(positions, tree)
        self.v += self.f
        if self.v.length() > config.MaxVel:
            self.v.scale_to_length(config.MaxVel)
        self.p += self.v

        # Wrap around screen
        if self.p.x < 0:
            self.p.x = config.MonitorSize[0]
        elif self.p.x > config.MonitorSize[0]:
            self.p.x = 0
        if self.p.y < 0:
            self.p.y = config.MonitorSize[1]
        elif self.p.y > config.MonitorSize[1]:
            self.p.y = 0

    def draw(self, screen):
        size = 10
        angle = self.v.angle_to(pg.Vector2(1, 0))
        triangle_surface = pg.Surface((size * 2, size * 2), pg.SRCALPHA)
        pg.draw.polygon(triangle_surface, color.red, [(size, 0), (0, size * 4), (size * 2, size * 4)])
        rotated_surface = pg.transform.rotate(triangle_surface, 270 + angle)
        rotated_rect = rotated_surface.get_rect(center=(self.p.x, self.p.y))
        screen.blit(rotated_surface, rotated_rect.topleft)

class Birds:
    def __init__(self, flock_size):
        self.flock = np.array([Bird(self) for _ in range(flock_size)], dtype=Bird)

    def add(self, count):
        new_birds = np.array([Bird(self) for _ in range(count)], dtype=Bird)
        self.flock = np.concatenate((self.flock, new_birds))

    def delete(self, count):
        if count >= len(self.flock):
            self.flock = np.array([], dtype=Bird)
        else:
            self.flock = self.flock[:-count]

    def update(self):
        positions = np.array([[bird.p.x, bird.p.y] for bird in self.flock])
        tree = KDTree(positions)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(lambda b: b.update(positions, tree), self.flock)

    def draw(self, screen):
        for b in self.flock:
            b.draw(screen)

if __name__ == "__main__":
    g = Game()
    b = Birds(config.BirdNumber)
    g.run(b)



    