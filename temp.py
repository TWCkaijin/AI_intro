# *-* coding: utf-8 *-*

import pygame as pg
from threading import Thread as th 
import numpy as np
import math
from numba import cuda
import os
import concurrent.futures


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYGAME_DETECT_AVX2"] = "1"
mother_board = None
device_board = None

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
        pg.init()
        
        self.running = True
        self.clock = pg.time.Clock()
        self.fps = config.FrameRate
        self.font = pg.font.Font(None, 24)
        self.board_tick = True

        #Mother Board config
        global mother_board, device_board
        mother_board = np.full(shape=(config.MonitorSize[0],config.MonitorSize[1],7),fill_value=-1) # 800x600 x 每單位資料(force x, force y, velocity x, velocity y)
        device_board = cuda.to_device(mother_board)
        self.update_monitor() #CUDA args config update

        self.flock_board = None



    def draw_status(self, birds):
        fps = self.clock.get_fps()
        bird_count = len(birds.flock)
        status = f"FPS: {fps:.2f} | Bird Count: {bird_count}"
        text_surface = self.font.render(status, True, color.black)
        screen.blit(text_surface, (10, 10))

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

            screen.fill(color.white)
            birds.update()
            self.draw_status(birds)
            pg.display.flip()
            self.clock.tick(self.fps)

        pg.quit()
    
    def update_cuda_config(self):
        self.config_arr = np.array([config.Sight, config.alignmentFactor, config.cohesionFactor, config.separationFactor],dtype=np.float32)
        self.board_tpb = (32,32) # threads per block
        self.board_bpg_x = math.ceil(device_board.shape[0]/self.tpb[0])  # blocks per grid x
        self.board_bpg_y = math.ceil(device_board.shape[1]/self.tpb[1])  # blocks per grid y
        self.board_bpg = (self.board_bpg_x,self.board_bpg_y)     # blocks per grid

    def update_game(self,cluster_obj):
        bt = not self.board_tick
        ############################################################################################################
        # CUDA FUNCTION
        @cuda.jit
        def cuda_MotherBoardUpdate(board,config_arr,cx,cy):
            x,y = cuda.grid(2)
            if board[x][y][0] == 0 and board[x][y][1] == 0:
                return
            r = (cx-x)**2 + (cy-y)**2
            if r==0:
                return
            if(r < config_arr[0]):
                board[x][y][2] += board[cx][cy][0]*config_arr[1] + cx*config_arr[2]
                board[x][y][3] += board[cx][cy][1]*config_arr[1] + cy*config_arr[2]
                board[x][y][4] += (-x/(r*r)) * config_arr[0]*config_arr[3]
                board[x][y][5] += (-y/(r*r)) * config_arr[0]*config_arr[3]
                board[x][y][6] += 1

        ############################################################################################################
        # CUDA FUNCTION
        @cuda.jit
        def update_cuda_pos(board,cx,cy):
            x,y = cuda.grid(2)
            if(x==cx and y==cy):
                temp = board[x][y].copy()
                """ might be a problem"""

                vx,vy = board[x][y][0],board[x][y][1]
                board[x][y] = board[x+vx][y+vy]
                board[x+vx][y+vy] = temp
        ############################################################################################################
        # CUDA Function
        @cuda.jit
        def update_flock_list(board,flock):
            i = cuda.grid(1)
            cx,cy = flock[i].p[0], flock[i].p[1]
            flock[i].p[0] = cx + board[cx][cy][0]
            flock[i].p[1] = cy + board[cx][cy][1]
        ############################################################################################################
        # change it into cuda

        device_flock = cuda.to_device(cluster_obj.flock_pos)
        flock_tpb = 256
        blockspergrid = (device_flock.size + (flock_tpb - 1)) // flock_tpb
        #device_inc_one[blockspergrid,threadsperblock](x_device)
        for bird in cluster_obj.flock:
            mother_board[int(bird.p[0]), int(bird.p[1])] = [bird.v[0], bird.v[1], 0, 0, 0, 0, 0] 
                                # velocity_x, velocity_y, force_x, force_y, separation_total_x, separation_total_y, counter
                                # ob          ob          cv       cv       sv                  sv                  sv    
                                # ob = object, cv = combined value, sv = single value       

        ############################################################################################################
        #call cuda function

        for b in cluster_obj.flock:
            cuda_MotherBoardUpdate[self.board_bpg,self.board_tpb](device_board, self.config_arr, int(b.p[0]),int(b.p[1]))


        device_flock = cuda.to_device(cluster_obj.flock_pos)
        flock_tpb = 256
        flock_bpg = (device_flock.size + (flock_tpb - 1)) // flock_tpb
        update_flock_list[blockspergrid,threadsperblock](x_device)
        ############################################################################################################
        cuda.synchronize()
        mother_board = device_board.copy_to_host()
        for bird in cluster_obj.flock:
            x,y = int(bird.p[0]),int(bird.p[1])
            if(mother_board[x,y][6] != 0):
                bird.f[0] = mother_board[x,y][2]/mother_board[x,y][6] # force_x / counter
                bird.f[1] = mother_board[x,y][3]/mother_board[x,y][6] # force_y / counter
            bird.f[0] += mother_board[x,y][4]
            bird.f[1] += mother_board[x,y][5]


class birds:
    def __init__(self,flock_size,main_root):
        self.flock = np.array([ bird(self) for _ in range(flock_size) ], dtype=bird)
        self.flock_pos = np.array([ self.flock[i].p for i in range(flock_size) ], dtype=pg.Vector2)
        self.root = main_root
        self.t = 0

    def delete(self, num=1):
        for _ in range(num):
            self.flock = np.delete(self.flock, -1)

    def add(self, num=1):
        for _ in range(num):
            self.flock = np.append(self.flock, bird(self))

    def update(self):
        self.root.update_game(self)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(lambda b: b.next_tick(), self.flock)
            


class bird(pg.sprite.Sprite):

    def __init__(self,pt:birds):
        super().__init__()
        self.p = pg.Vector2(np.random.uniform(0, config.MonitorSize[0]), np.random.uniform(0, config.MonitorSize[1]))  # 隨機 x, y
        self.v = pg.Vector2(np.random.uniform(-config.InitVel, config.InitVel), np.random.uniform(-config.InitVel, config.InitVel))  # 隨機 x, y
        self.f = pg.Vector2(0, 0)
        self.pattern = pt


    def next_tick(self):
        #input()
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


    def draw(self):
        size = 10  
        angle = self.v.angle_to(pg.Vector2(1, 0))
        triangle_surface = pg.Surface((size * 2, size * 2), pg.SRCALPHA)
        pg.draw.polygon(triangle_surface, (255, 0, 0), [(size, 0), (0, size * 4), (size * 2, size * 4)])
        rotated_surface = pg.transform.rotate(triangle_surface, 270 + angle)
        rotated_rect = rotated_surface.get_rect(center=(self.p.x, self.p.y))
        screen.blit(rotated_surface, rotated_rect.topleft)




if __name__ == '__main__':
    cuda.select_device(0)
    screen = pg.display.set_mode(config.MonitorSize, pg.RESIZABLE)
    g = game()
    b = birds(config.BirdNumber, g)



    g = game()
    b = birds(config.BirdNumber,g)
    g.run(b)


    


