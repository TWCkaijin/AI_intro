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
    #畫面PIXEL
    MonitorSize = (800, 600)
    #畫面PIXEL暫存
    tempMonitorSize = (800, 600)

    #幀率
    FrameRate = 120
    
    #最大速度
    MaxVel = 3
    #最小速度
    MinVel = 1
    #初始速度
    InitVel = 3
    
    #鳥數PIXEL比例
    flock_ratio = 1/2500

    ###################
    #### 可調整參數 ####
    #鳥數
    BirdNumber = 150
    #視野
    Sight = 30
    #對齊力參數
    alignmentFactor = 0.4
    #聚集力參數
    cohesionFactor = 0.2
    #分離力參數
    separationFactor = 1.5
    ####################

class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, start_val, label,show_float=True):
        self.rect = pg.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.value = start_val
        self.label = label
        self.dragging = False
        self.show_float = show_float

    def handle_event(self, event): 
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pg.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pg.MOUSEMOTION:
            if self.dragging:
                self.value = (event.pos[0] - self.rect.x) / self.rect.w * (self.max_val - self.min_val) + self.min_val
                self.value = max(self.min_val, min(self.value, self.max_val))

    def draw(self, screen):
        pg.draw.rect(screen, color.black, self.rect, 2)
        pg.draw.rect(screen, color.red, (self.rect.x, self.rect.y, (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.w, self.rect.h))
        font = pg.font.Font(None, 24)
        if self.show_float:
            text = font.render(f"{self.label}: {self.value:.2f}", True, color.black)
        else:
            text = font.render(f"{self.label}: {int(self.value)}", True, color.black)
        screen.blit(text, (self.rect.x, self.rect.y - 15))

class Game:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode(config.MonitorSize, pg.RESIZABLE)
        self.running = True
        self.clock = pg.time.Clock()
        self.fps = config.FrameRate
        self.font = pg.font.Font(None, 24)
        self.sliders = [
            Slider(10, 50, 100, 10, -0.1, 1.0, config.alignmentFactor, "Alignment"),
            Slider(10, 80, 100, 10, -1.0, 1.0, config.cohesionFactor, "Cohesion"),
            Slider(10, 110, 100, 10, 0, 5.0, config.separationFactor, "Separation"),
            Slider(10, 140, 100, 10, 0, 80, config.Sight, "Bird Sight",show_float=False),
            Slider(10, 170, 100, 10, 0, 800, config.BirdNumber, "Bird Number",show_float=False)
        ]

    def draw_status(self, birds):
        fps = self.clock.get_fps()
        bird_count = len(birds.flock)
        status = f"FPS: {fps:.2f} | Bird Count: {bird_count}"
        text_surface = self.font.render(status, True, color.black)
        self.screen.blit(text_surface, (10, 10))

    def run(self, birds):
        temp = config.MonitorSize[0] * config.MonitorSize[1]
        while self.running:
            resize = False
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                elif event.type == pg.VIDEORESIZE:
                    resize = True
                    config.tempMonitorSize = (event.w, event.h)
                for slider in self.sliders:
                    slider.handle_event(event)

            try:
                self.screen.fill(color.white)
                birds.update()
                birds.draw(self.screen)
                self.draw_status(birds)
                for slider in self.sliders:
                    slider.draw(self.screen)
                pg.display.flip()
                self.clock.tick(self.fps)
            except Exception as e:
                print(e)
                self.clock.tick(self.fps)

            if resize:
                """ new_size = config.tempMonitorSize[0] * config.tempMonitorSize[1]
                if new_size > temp:
                    birds.add(int((new_size - temp) * config.flock_ratio))
                else:
                    birds.delete(int((temp - new_size) * config.flock_ratio))
                temp = new_size """
                config.MonitorSize = config.tempMonitorSize

            # config update
            config.alignmentFactor = self.sliders[0].value
            config.cohesionFactor = self.sliders[1].value
            config.separationFactor = self.sliders[2].value
            config.Sight = self.sliders[3].value
            if (self.sliders[-1].value>config.BirdNumber):
                birds.add(int(self.sliders[-1].value-config.BirdNumber))
                config.BirdNumber = int(self.sliders[-1].value)
            elif (self.sliders[-1].value<config.BirdNumber):
                birds.delete(int(config.BirdNumber-self.sliders[-1].value))
                config.BirdNumber = int(self.sliders[-1].value)

        pg.quit()

class Bird:
    def __init__(self, pattern):
        self.pattern = pattern
        self.p = pg.Vector2(random.uniform(0, config.MonitorSize[0]), random.uniform(0, config.MonitorSize[1]))
        self.v = pg.Vector2(random.uniform(-config.InitVel, config.InitVel), random.uniform(-config.InitVel, config.InitVel))
        self.f = pg.Vector2(0, 0)

    def get_distance(self, bird) -> pg.Vector2:
        return bird.p - self.p  #曼哈頓距離

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
                separation_total += -np.array([vector.x, vector.y]) / ((distance if distance else 1) ** 2) * config.Sight

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

        if self.p.x < 0:
            self.p.x = config.MonitorSize[0]
        elif self.p.x > config.MonitorSize[0]:
            self.p.x = 0
        if self.p.y < 0:
            self.p.y = config.MonitorSize[1]
        elif self.p.y > config.MonitorSize[1]:
            self.p.y = 0

    def draw(self, screen, color):
        size = 10
        angle = self.v.angle_to(pg.Vector2(1, 0))
        triangle_surface = pg.Surface((size * 2, size * 2), pg.SRCALPHA)
        pg.draw.polygon(triangle_surface, color, [(size, 0), (0, size * 4), (size * 2, size * 4)])
        rotated_surface = pg.transform.rotate(triangle_surface, 270 + angle)
        rotated_rect = rotated_surface.get_rect(center=(self.p.x, self.p.y))
        screen.blit(rotated_surface, rotated_rect.topleft)

class Birds:
    def __init__(self, flock_size, flock_color=color.red):
        self.flock = np.array([Bird(self) for _ in range(flock_size)], dtype=Bird)
        self.flock_color = flock_color
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
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(lambda b: b.draw(screen,self.flock_color), self.flock)

if __name__ == "__main__":
    g = Game()
    Specie_A = Birds(config.BirdNumber,flock_color=color.red)
    #Specie_B = Birds(config.BirdNumber,flock_color=color.blue)
    g.run(Specie_A)