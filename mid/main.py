# *-* coding: utf-8 *-*

import pygame as pg
from threading import Thread as th 
import random
import math
import numpy as np
import time


class color:
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)

class config:
    MonitorSize = (800, 600)
    FrameRate = 60
    MaxVel = 3
    MinVel = 1
    Sight = 30

    BirdNumber = 150
    flock_ratio = 1/2500

    InitVel = 3
    alignmentFactor = 0.3
    cohesionFactor = 0.1
    separationFactor = 1




class game: 
    def __init__(self):
        self.screen = pg.display.set_mode(config.MonitorSize, pg.RESIZABLE)
        self.running = True
        self.clock = pg.time.Clock()
        self.fps = config.FrameRate




class birds:
    def __init__(self,flock_size):
        self.flock = np.array([ bird(self) for _ in range(flock_size) ], dtype=bird)
        #self.flock_pos = np.array([ pg.Vector2(self.flock[i].p) for i in range(flock_size) ], dtype=pg.Vector2)

    def delete(self, num=1):
        for _ in range(num):
            self.flock = np.delete(self.flock, -1)

    def add(self, num=1):
        for _ in range(num):
            self.flock = np.append(self.flock, bird(self))

    def update(self) -> bool :
        for b in self.flock:
            b.next_tick()
        return True


class bird(pg.sprite.Sprite):

    def __init__(self,pt:birds):
        super().__init__()
        self.p = pg.Vector2(np.random.uniform(0, config.MonitorSize[0]), np.random.uniform(0, config.MonitorSize[1]))  # 隨機 x, y
        self.v = pg.Vector2(np.random.uniform(-config.InitVel, config.InitVel), np.random.uniform(-config.InitVel, config.InitVel))  # 隨機 x, y
        self.f = pg.Vector2(0, 0)
        self.pattern = pt
        

    
    def next_tick(self):
        self.generate_force()
        self.v += self.f
        self.limit_speed()
        self.p += self.v
        self.handle_boundaries()
        self.draw()
    
    def limit_speed(self):
        if self.v.length() > config.MaxVel:
            self.v.scale_to_length(config.MaxVel)

    def handle_boundaries(self):
        stage_size = pg.Vector2(config.MonitorSize)
        if self.p.x < 0:
            self.p.x += stage_size.x
        elif self.p.x > stage_size.x:
            self.p.x -= stage_size.x

        if self.p.y < 0:
            self.p.y += stage_size.y
        elif self.p.y > stage_size.y:
            self.p.y -= stage_size.y

    def update_sprite(self):
        self.rect.x = self.p.x
        self.rect.y = self.p.y
        self.image = pg.transform.rotate(self.image, -self.v.angle_to(pg.Vector2(1, 0)))

    def get_distance(self, other):
        vector = other.p - self.p
        stage_size = pg.Vector2(config.MonitorSize)

        if vector.x > stage_size.x / 2:
            vector.x -= stage_size.x
        elif vector.x < -stage_size.x / 2:
            vector.x += stage_size.x

        if vector.y > stage_size.y / 2:
            vector.y -= stage_size.y
        elif vector.y < -stage_size.y / 2:
            vector.y += stage_size.y

        return vector

    def generate_force(self):
        self.f.x = 0
        self.f.y = 0

        alignment_total = pg.Vector2(0, 0)
        cohesion_total = pg.Vector2(0, 0)
        separation_total = pg.Vector2(0, 0)
        total = 0

        for bird in self.pattern.flock:
            vector = self.get_distance(bird)
            distance = vector.length()

            if distance < config.Sight and bird != self:
                total += 1

                alignment_total.x += bird.v.x
                alignment_total.y += bird.v.y
                cohesion_total.x += vector.x
                cohesion_total.y += vector.y
                separation_total.x += -vector.x / distance / distance * config.Sight
                separation_total.y += -vector.y / distance / distance * config.Sight

        if total != 0:
            self.f.x += alignment_total.x / total * config.alignmentFactor
            self.f.y += alignment_total.y / total * config.alignmentFactor
            self.f.x += cohesion_total.x / total * config.cohesionFactor
            self.f.y += cohesion_total.y / total * config.cohesionFactor
            self.f.x += separation_total.x * config.separationFactor
            self.f.y += separation_total.y * config.separationFactor

    def draw(self):
        size = 10  

        # 計算速度向量的角度
        angle = self.v.angle_to(pg.Vector2(1, 0))

        # 創建一個新的 Surface 來繪製三角形
        triangle_surface = pg.Surface((size * 2, size * 2), pg.SRCALPHA)
        pg.draw.polygon(triangle_surface, (255, 0, 0), [(size, 0), (0, size * 4), (size * 2, size * 4)])

        # 旋轉三角形
        rotated_surface = pg.transform.rotate(triangle_surface, 270+angle)

        # 獲取旋轉後的矩形區域
        rotated_rect = rotated_surface.get_rect(center=(self.p.x, self.p.y))

        # 在屏幕上繪製旋轉後的三角形
        screen.blit(rotated_surface, rotated_rect.topleft)

if __name__ == '__main__':
    pg.init()
    screen = pg.display.set_mode((800, 600))
    pg.display.set_caption("Bird Simulation")

    g = game()
    b = birds(config.BirdNumber)



    while g.running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.VIDEORESIZE:
                temp  = config.MonitorSize[0]*config.MonitorSize[1]
                screen = pg.display.set_mode((event.w, event.h), pg.RESIZABLE)
                config.MonitorSize = (event.w, event.h)
                new_size = event.w*event.h
                if(new_size > temp):
                    b.add(int((new_size-temp)*config.flock_ratio))
                else:
                    b.delete(int((temp-new_size)*config.flock_ratio))

        screen.fill((255, 255, 255))
        b.update()

        # 更新顯示
        pg.display.flip()
        
        # 控制幀率
        g.clock.tick(config.FrameRate)


    pg.quit()


    


