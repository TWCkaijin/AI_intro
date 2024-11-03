import numpy as np
from scipy.spatial import cKDTree
import concurrent.futures
import pygame as pg
import random
import math
class Color:
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)

class config:
    #畫面大小
    MonitorSize = (800, 600)
    #畫面大小暫存
    tempMonitorSize = (800, 600)

    #幀率
    FrameRate = 120
    
    #最大速度
    MaxVel = 3
    #最小速度
    MinVel = 1
    #初始速度
    InitVel = 5
    
    #鳥數PIXEL比例
    flock_ratio = 1/2500

    ###################
    #### 可調整參數 ####
    #鳥數
    BirdNumber = 150
    #視野
    Sight = 30
    #對齊力參數
    alignmentFactor = 2
    #聚集力參數
    cohesionFactor = 1
    #分離力參數
    separationFactor = 1.5
    #牆壁避免力參數
    wallAvoidanceFactor = 0.5
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
        pg.draw.rect(screen, Color.white, self.rect, 2)
        pg.draw.rect(screen, Color.red, (self.rect.x, self.rect.y, (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.w, self.rect.h))
        font = pg.font.Font(None, 18)
        if self.show_float:
            text = font.render(f"{self.label}: {self.value:.2f}", True, Color.white)
        else:
            text = font.render(f"{self.label}: {int(self.value)}", True, Color.white)
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
            Slider(10, 50, 100, 10, -1.0, 3.0, config.alignmentFactor, "Alignment"),
            Slider(10, 80, 100, 10, -2.0, 2.0, config.cohesionFactor, "Cohesion"),
            Slider(10, 110, 100, 10, 0, 5.0, config.separationFactor, "Separation"),
            Slider(10, 140, 100, 10, 0, 80, config.Sight, "Bird Sight",show_float=False),
            Slider(10, 170, 100, 10, 0, 8, config.MaxVel, "speed"),
            Slider(10, 210, 100, 10, 0, 500, config.BirdNumber, "Bird Number",show_float=False)
        ]
        self.walls = []  # 儲存牆壁的列表
        self.drawing_wall = False  # 是否正在繪製牆壁
        self.wall_start_pos = None  # 牆壁的起始位置

    def draw_status(self, birds):
        fps = self.clock.get_fps()
        bird_count = len(birds.flock)
        status = f"FPS: {fps:.2f} | Bird Count: {bird_count}"
        text_surface = self.font.render(status, True, Color.white)
        self.screen.blit(text_surface, (10, 10))

    def draw_walls(self):
        for wall in self.walls:
            start, end = wall
            pg.draw.line(self.screen, Color.green, start, end, width=5)

    def run(self, birds):
        while self.running:
            resize = False
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False

                elif event.type == pg.VIDEORESIZE:
                    resize = True
                    config.tempMonitorSize = (event.w, event.h)

                elif event.type == pg.MOUSEBUTTONDOWN:
                    """ for bird in birds.flock:
                        if bird.is_clicked(event.pos):
                            bird.color = Color.red """
                    
                    if event.button == 1:  # 左鍵繪製牆壁
                        self.drawing_wall = True
                        self.wall_start_pos = event.pos
                    elif event.button == 3:  # 右鍵刪除牆壁
                        self.delete_wall_at(event.pos)

                elif event.type == pg.MOUSEBUTTONUP:
                    if event.button == 1 and self.drawing_wall:
                        self.drawing_wall = False
                        wall_end_pos = event.pos
                        self.walls.append((self.wall_start_pos, wall_end_pos))

                for slider in self.sliders:
                    slider.handle_event(event) 

            try:
                self.screen.fill(Color.black)
                birds.update(self.walls)
                birds.draw(self.screen)
                self.draw_walls()
                self.draw_status(birds)

                for slider in self.sliders:
                    slider.draw(self.screen)

                pg.display.flip()
                self.clock.tick(self.fps)
            except Exception as e:
                print(e)
                self.clock.tick(self.fps)

            if resize:
                config.MonitorSize = config.tempMonitorSize

            #################################
            ########### 參數update ##########
            config.alignmentFactor = self.sliders[0].value
            config.cohesionFactor = self.sliders[1].value
            config.separationFactor = self.sliders[2].value
            config.Sight = self.sliders[3].value
            config.MaxVel = self.sliders[4].value
            if (self.sliders[-1].value>config.BirdNumber):
                birds.add(int(self.sliders[-1].value-config.BirdNumber))
                config.BirdNumber = int(self.sliders[-1].value)
            elif (self.sliders[-1].value<config.BirdNumber):
                birds.delete(int(config.BirdNumber-self.sliders[-1].value))
                config.BirdNumber = int(self.sliders[-1].value)
            ########### 參數update ##########
            #################################
        pg.quit()

    def delete_wall_at(self, pos):
        for wall in self.walls:
            if self.is_point_near_line(pos, wall[0], wall[1], threshold=10):
                self.walls.remove(wall)
                break

def is_point_near_line(self, point, start, end, threshold):
    # 檢查點是否在線段附近，使用點到線段的距離公式
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:
        # 線段起點和終點相同
        distance = math.hypot(px - x1, py - y1)
        return distance <= threshold
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    distance = math.hypot(px - nearest_x, py - nearest_y)
    return distance <= threshold

class Bird:
    def __init__(self, pattern,color):
        self.pattern = pattern
        self.p = pg.Vector2(random.uniform(0, config.MonitorSize[0]), random.uniform(0, config.MonitorSize[1]))
        self.v = pg.Vector2(random.uniform(-config.InitVel, config.InitVel), random.uniform(-config.InitVel, config.InitVel))
        self.f = pg.Vector2(0, 0)
        self.color = color

    def get_distance(self, other_pos, offset):
        dx = other_pos[0] + offset[0] * config.MonitorSize[0] - self.p.x
        dy = other_pos[1] + offset[1] * config.MonitorSize[1] - self.p.y
        return pg.Vector2(dx, dy)

    def generate_force(self, positions, offsets, tree, walls):
        self.f.x = 0
        self.f.y = 0

        # 查詢鄰居
        query_point = [self.p.x, self.p.y]
        indices = tree.query_ball_point(query_point, config.Sight)

        alignment_total = pg.Vector2(0, 0)
        cohesion_total = pg.Vector2(0, 0)
        separation_total = pg.Vector2(0, 0)
        total = 0

        for idx in indices:
            bird_idx = idx % len(self.pattern.flock)  # 真實的鳥索引
            bird = self.pattern.flock[bird_idx]
            if bird == self:
                continue

            offset = offsets[idx]  # 平移偏移量
            vector = self.get_distance([bird.p.x, bird.p.y], offset)
            distance = vector.length()

            if distance < config.Sight and distance > 0:
                total += 1

                alignment_total += bird.v
                cohesion_total += vector
                separation_total += -vector / (distance ** 2) * config.Sight

        if total > 0:
            alignment_avg = alignment_total / total
            cohesion_avg = cohesion_total / total

            self.f += alignment_avg * config.alignmentFactor
            self.f += cohesion_avg * config.cohesionFactor
            self.f += separation_total * config.separationFactor

        wall_avoidance_force = self.avoid_walls(walls)
        self.f += wall_avoidance_force * config.wallAvoidanceFactor

    def update(self, positions, offsets, tree, walls):
        self.generate_force(positions, offsets, tree, walls)
        self.v += self.f
        if self.v.length() > config.MaxVel:
            self.v.scale_to_length(config.MaxVel)
        elif self.v.length() < config.MinVel:
            self.v.scale_to_length(config.MinVel)
        self.p += self.v

        # 環繞邊界
        self.p.x %= config.MonitorSize[0]
        self.p.y %= config.MonitorSize[1]

    # 單一鳥繪製
    def draw(self, screen):
        size = 10
        angle = self.v.angle_to(pg.Vector2(1, 0))
        triangle_surface = pg.Surface((size * 2, size * 2), pg.SRCALPHA)
        pg.draw.polygon(triangle_surface, self.color, [(size, 0), (0, size * 4), (size * 2, size * 4)])
        rotated_surface = pg.transform.rotate(triangle_surface, 270 + angle)
        rotated_rect = rotated_surface.get_rect(center=(self.p.x, self.p.y))
        screen.blit(rotated_surface, rotated_rect.topleft)

    def is_clicked(self, mouse_pos):
        size = 10
        rect = pg.Rect(self.p.x - size, self.p.y - size, size * 2, size * 2)
        return rect.collidepoint(mouse_pos)
    
    def avoid_walls(self, walls):
        avoid_force = pg.Vector2(0, 0)
        for wall in walls:
            start, end = wall
            nearest_point = self.get_nearest_point_on_line(self.p, start, end)
            distance = self.p.distance_to(nearest_point)
            if distance < config.Sight:
                # 計算避開牆壁的力，與牆壁垂直，方向遠離牆壁
                direction = self.p - nearest_point
                if direction.length() > 0:
                    direction.normalize_ip()
                    avoid_force += direction * (config.Sight - distance)
        return avoid_force

    def get_nearest_point_on_line(self, point, start, end):
        # 計算點到線段的最近點
        line_vec = pg.Vector2(end) - pg.Vector2(start)
        point_vec = pg.Vector2(point) - pg.Vector2(start)
        line_len = line_vec.length()
        line_unitvec = line_vec.normalize()
        proj_length = point_vec.dot(line_unitvec)
        proj_length = max(0, min(line_len, proj_length))
        nearest = pg.Vector2(start) + line_unitvec * proj_length
        return nearest

class Birds:
    def __init__(self, flock_size, flock_Color=Color.white):
        self.flock = np.array([Bird(self,flock_Color) for _ in range(flock_size)], dtype=Bird)
        self.color = flock_Color
    def add(self, count):
        new_birds = np.array([Bird(self,self.color) for _ in range(count)], dtype=Bird)
        self.flock = np.concatenate((self.flock, new_birds))

    def delete(self, count):
        if count >= len(self.flock):
            self.flock = np.array([], dtype=Bird)
        else:
            self.flock = self.flock[:-count]

    def update(self,walls):
        positions = np.array([[bird.p.x, bird.p.y] for bird in self.flock])
        N = len(self.flock)

        # 螢幕外虛擬位置
        offsets = np.array([[0, 0],
                            [1, 0], [-1, 0],
                            [0, 1], [0, -1],
                            [1, 1], [-1, -1],
                            [1, -1], [-1, 1]])
        all_positions = []
        all_offsets = []

        for offset in offsets:
            shifted_positions = positions + offset * np.array(config.MonitorSize)
            all_positions.append(shifted_positions)
            all_offsets.extend([offset] * N)

        all_positions = np.vstack(all_positions)
        all_offsets = np.array(all_offsets)

        tree = cKDTree(all_positions)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(lambda b: b.update(all_positions, all_offsets, tree,  walls), self.flock)

    # 鳥群繪製
    def draw(self, screen):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(lambda b: b.draw(screen), self.flock)

if __name__ == "__main__":
    g = Game()
    Specie_A = Birds(config.BirdNumber,Color.white)
    #Specie_B = Birds(config.BirdNumber,flock_Color=Color.blue)
    g.run(Specie_A)