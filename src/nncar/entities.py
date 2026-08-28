import pygame,math,time,pickle,random
from nncar import window as w
from nncar.neural_network import forward_propagation,random_normal,Network,number_of_cars
from nncar import assets

class Car:

    def __init__(self):
        self.offset_x = 0
        self.offset_y = 0
        self.width = 80
        self.height = 140
        self.velocity = 0
        self.rotational_velocity = 0
        self.max_velocity = 12
        self.angle = 0
        self.rotational_sensitivity = 15
        self.collision_cooldown = False
        self.bouncing_multiplier = 0.5
        self.checkpoints = [Checkpoint(450,-355,1030,-355),
                            Checkpoint(1810,-690,1810,-120),
                            Checkpoint(1765,375,2360,375),
                            Checkpoint(3395,400,3395,960),
                            Checkpoint(2840,-540,2840,-265),
                            Checkpoint(3370,-100,3930,-100),
                            Checkpoint(3365,1440,3935,1440),
                            Checkpoint(2325,1455,2325,2005),
                            Checkpoint(1050,1205,1050,1755),
                            Checkpoint(465,665,1025,665)]
        self.checkpoints_passed = 0
        self.last_checkpoint = 0
        self.laps = 0
        self.collision_event = Event(0.01)

    def draw(self):
        rotated_image = pygame.transform.rotate(self.image,self.angle)
        self.centre = rotated_image.get_rect(center=self.image.get_rect(topleft=(self.x+player.offset_x-self.offset_x,self.y+player.offset_y-self.offset_y)).center)
        w.window.blit(rotated_image,self.centre.topleft)
    
    def bounce(self):
        if not self.collision_cooldown:
            self.collision_cooldown = True
            if self.velocity > 0:
                self.velocity = min(-self.bouncing_multiplier*self.velocity,-3)
            else:
                self.velocity = max(-self.bouncing_multiplier*self.velocity,3)

    def steering(self):
        if self.velocity > 0:
            self.rotational_velocity = self.rotational_sensitivity / (self.velocity+1)
            self.rotational_velocity = min(self.rotational_velocity,self.velocity/4)
        else:
            self.rotational_velocity = self.rotational_sensitivity / (self.velocity-1)
            self.rotational_velocity = max(self.rotational_velocity,self.velocity/4)

    def collide(self):
        mask = pygame.mask.from_surface(self.image)
        offset = (int(self.x-track.x-self.offset_x),int(self.y-track.y-self.offset_y))
        collision = None
        border_number = 0
        while collision == None and border_number < 3:
            collision = track.mask[border_number].overlap(mask,offset)
            border_number += 1
        return collision

    def wrong_way(self):
        if self.x > track.line_x1 + self.offset_x and self.x < track.line_x2 + self.offset_x:
            if self.y + self.height > track.line_y1 + self.offset_y and self.y + self.height + self.velocity * math.cos(self.angle*math.pi/180) < track.line_y2 + self.offset_y:
                return True

    def check_checkpoints(self):
        for checkpoint in self.checkpoints:
            checkpoint.pass_checkpoint(self)

    def reset_checkpoints(self):
        if self.x > track.line_x1 + self.offset_x and self.x < track.line_x2 + self.offset_x:
            if self.y < track.line_y1 + self.offset_y and self.y + self.velocity * math.cos(self.angle*math.pi/180) > track.line_y2 + self.offset_y:
                self.collisions = 0
                self.checkpoints_passed = 0
                self.laps += 1
                self.checkpoints = [Checkpoint(450,-355,1030,-355),
                                    Checkpoint(1810,-690,1810,-120),
                                    Checkpoint(1765,375,2360,375),
                                    Checkpoint(3395,400,3395,960),
                                    Checkpoint(2840,-540,2840,-265),
                                    Checkpoint(3370,-100,3930,-100),
                                    Checkpoint(3365,1440,3935,1440),
                                    Checkpoint(2325,1455,2325,2005),
                                    Checkpoint(1050,1205,1050,1755),
                                    Checkpoint(465,665,1025,665)]


class PlayerCar(Car):
    def __init__(self):
        super().__init__()
        self.x = w.window_width//2
        self.y = w.window_height//2
        self.drift_angle = 0
        self.end_drift = False
        self.image = CARS[selected][0]
        self.image = pygame.transform.scale(self.image,(80,140))
        self.drift_event = Event(1.5)
        self.type = "player"
        self.placement = 0

    def movement(self):
        keys = pygame.key.get_pressed()
        if self.wrong_way():
            self.bounce()
        elif self.collide() == None:
            if self.collision_cooldown:
                if self.collision_event.check():
                    self.collision_cooldown = False
            if not keys[pygame.K_SPACE]:
                self.bouncing_multiplier = 0.5
                self.velocity = round(self.velocity,1)
                self.move(keys)
                self.drift_angle = self.angle
                self.end_drift = False
        else:
            self.bounce()
        if keys[pygame.K_SPACE]:
            self.bouncing_multiplier = 0.25
            self.drift()
        else:
            self.offset_x += self.velocity * math.sin(self.angle*math.pi/180)
            self.offset_y += self.velocity * math.cos(self.angle*math.pi/180)
        self.steering()
        if keys[pygame.K_LEFT]:
            self.angle += self.rotational_velocity
        if keys[pygame.K_RIGHT]:
            self.angle -= self.rotational_velocity

    def move(self,keys):
        slowing = False
        if keys[pygame.K_UP]:
            if self.velocity < self.max_velocity:
                self.velocity += 0.1
        elif keys[pygame.K_DOWN]:
            if self.velocity > -self.max_velocity//2:
                self.velocity -= 0.1
        else:
            slowing = True
        if slowing:
            if self.velocity > 0:
                self.velocity -= 0.1
            elif self.velocity < 0:
                self.velocity += 0.1

    def drift(self):
        if self.drift_event.check():
            self.end_drift = True
        self.offset_x += (self.velocity * math.sin(self.drift_angle*math.pi/180)) / 2 + (self.velocity * math.sin(self.angle*math.pi/180)) / 2
        self.offset_y += (self.velocity * math.cos(self.drift_angle*math.pi/180)) / 2 + (self.velocity * math.cos(self.angle*math.pi/180)) / 2
        if self.end_drift:
            self.velocity = round(self.velocity,1)
            if self.velocity > 0:
                self.velocity -= 0.1
            elif self.velocity < 0:
                self.velocity += 0.1
        else:
            if self.velocity > 0:
                self.velocity -= 0.02
            elif self.velocity < 0:
                self.velocity += 0.02

class NPC(Car):
    start_positions = []

    def __init__(self,image,laps,network):
        super().__init__()
        try:
            position = random.choice(NPC.start_positions)
            NPC.start_positions.remove(position)
        except:
            position = [610,490]
        self.x,self.y = position
        self.image = image
        self.image = pygame.transform.scale(self.image,(80,140))
        self.collisions = 0
        self.network = network
        self.accelerate = 0
        self.turn = 0
        self.laps = laps
        self.type = "NPC"
        #: Whether this network expects sensor inputs scaled to [0,1].
        #: Models predating normalisation are loaded with this False.
        self.normalise_inputs = False
        self.sensors = [Sensor(self.x+self.width//2,self.y+self.height//2,self.angle-90,500),
                        Sensor(self.x+self.width//2,self.y+self.height//2,self.angle-45,600),
                        Sensor(self.x+self.width//2,self.y+self.height//2,self.angle,700),
                        Sensor(self.x+self.width//2,self.y+self.height//2,self.angle+45,600),
                        Sensor(self.x+self.width//2,self.y+self.height//2,self.angle+90,500)]

    def move(self):
        self.accelerate,self.turn = forward_propagation(self)
        self.accelerate += random_normal() * 0.15
        self.turn += random_normal() * 0.15
        if self.wrong_way():
            self.bounce()
            self.collisions += 1
        elif self.collide() == None:
            if self.collision_cooldown:
                if self.collision_event.check():
                    self.collision_cooldown = False
            if abs(self.velocity) < self.max_velocity:
                self.velocity += 0.1 * self.accelerate
        else:
            self.bounce()
            self.collisions += 1
        self.steering()
        self.angle -= self.rotational_velocity * self.turn
        self.x -= self.velocity * math.sin(self.angle*math.pi/180)
        self.y -= self.velocity * math.cos(self.angle*math.pi/180)

    def update_sensors(self):
        for sensor in self.sensors:
            sensor.update(self.velocity,self.angle,self.rotational_velocity,self.turn)
        self.inputs = [[self.sensors[0].distance()],
                       [self.sensors[1].distance()],
                       [self.sensors[2].distance()],
                       [self.sensors[3].distance()],
                       [self.sensors[4].distance()],
                       [self.velocity]]
        
class Track:
    def __init__(self,lap_number):
        self.x = 400
        self.y = -1000
        self.image = pygame.image.load(assets.image("track.png"))
        self.border = [pygame.image.load(path) for path in assets.border_paths()]
        self.mask = [pygame.mask.from_surface(self.border[0]),pygame.mask.from_surface(self.border[1]),pygame.mask.from_surface(self.border[2])]
        self.lap_number = lap_number
        self.leaderboard = []
        self.line_x1,self.line_y1 = 450,665
        self.line_x2,self.line_y2 = 1050,665

    def draw(self):
         w.window.blit(self.image,(self.x+player.offset_x,self.y+player.offset_y))

    def get_pixel_alpha(self,x,y):
        try:
            for border in self.border:
                pixel_colour = border.get_at((int(x),int(y)))
                alpha_value = pixel_colour[3]
                if alpha_value != 0:
                    return True
            return False
        except:
            return False

class Sensor:
    def __init__(self,car_x,car_y,angle,length):
        self.x1 = self.x2 = car_x
        self.y1 = self.y2 = car_y
        self.length = length
        self.angle = angle

    def update(self,car_velocity,car_angle,car_rotational_velocity,turn):
        self.angle -= car_rotational_velocity * turn
        self.x1 -= car_velocity * math.sin(car_angle*math.pi/180)
        self.y1 -= car_velocity * math.cos(car_angle*math.pi/180)
        self.x2 = self.x1
        self.y2 = self.y1

    def new_coordinates(self):
        self.x2 -= 5 * math.sin(math.radians(self.angle))
        self.y2 -= 5 * math.cos(math.radians(self.angle))

    def collide(self):
        if track.get_pixel_alpha(self.x2-track.x,self.y2-track.y):
            return math.sqrt((self.x2-self.x1)**2 + (self.y2-self.y1)**2)
        if math.sqrt((self.x2-self.x1)**2 + (self.y2-self.y1)**2) > self.length:
            return self.length
        self.new_coordinates()
        return None
    
    def distance(self):
        distance = self.collide()
        while distance == None:
            distance = self.collide()
        return distance

class Checkpoint:
    def __init__(self,x1,y1,x2,y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        
    def pass_checkpoint(self,car):
        if self.y1 == self.y2:
            if car.x > self.x1 + car.offset_x and car.x < self.x2 + car.offset_x:
                if car.y < self.y1 + car.offset_y and car.y + car.velocity * math.cos(car.angle*math.pi/180) > self.y1 + car.offset_y or car.y > self.y1 + car.offset_y and car.y + car.velocity * math.cos(car.angle*math.pi/180) < self.y1 + car.offset_y:
                    car.last_checkpoint = time.time() - START_TIME
                    car.checkpoints_passed += 1
                    car.checkpoints.remove(self)
        elif self.x1 == self.x2:
            if car.y > self.y1 + car.offset_y and car.y < self.y2 + car.offset_y:
                if car.x < self.x1 + car.offset_x and car.x + car.velocity * math.sin(car.angle*math.pi/180) > self.x1 + car.offset_x or car.x > self.x1 + car.offset_x and car.x + car.velocity * math.sin(car.angle*math.pi/180) < self.x1 + car.offset_x:
                    car.last_checkpoint = time.time() - START_TIME
                    car.checkpoints_passed += 1
                    car.checkpoints.remove(self)

class Event:
    def __init__(self, reset_time):
        self.start_time = time.time()
        self.reset_time = reset_time
    
    def check(self):
        if time.time() - self.start_time > self.reset_time:
            self.start_time = time.time()
            return True
        else:
            return False
        
class Button:
    cursor_image = pygame.image.load(assets.image("cursor.png"))
    cursor_image = pygame.transform.scale(cursor_image,(32,32))

    def __init__(self,x,y,width,height,image):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.image = pygame.image.load(assets.image(image + "button.png"))
        self.image = pygame.transform.scale(self.image,(width,height))

    def update(self):
        w.window.blit(self.image,(self.x,self.y))
        
    def hover(self,mouse_pos):
        if mouse_pos[0] in range(self.x,self.x+self.width) and mouse_pos[1] in range(self.y,self.y+self.height):
            pygame.mouse.set_visible(False)
            w.window.blit(Button.cursor_image,(mouse_pos[0]-Button.cursor_image.get_width()//2,mouse_pos[1]-Button.cursor_image.get_height()//2))
            return True

class Action_Button(Button):
    def __init__(self,x,y,width,height,image,function):
        super().__init__(x,y,width,height,image)
        self.function = function
    
    def pressed(self,mouse_pos):
        if mouse_pos[0] in range(self.x,self.x+self.width) and mouse_pos[1] in range(self.y,self.y+self.height):
            self.function()

class Option_Button(Button):
    def __init__(self,x,y,width,height,image,new_variable):
        super().__init__(x,y,width,height,image)
        self.new_variable = new_variable

    def pressed(self,mouse_pos):
        if mouse_pos[0] in range(self.x,self.x+self.width) and mouse_pos[1] in range(self.y,self.y+self.height):
            return self.new_variable

START_TIME = time.time()
#: Purchasable car skins, paired with their shop price.
CAR_SKINS = [("red.png",0),("yellow.png",100),("orange.png",200),("lightgrey.png",300),
             ("white.png",400),("darkgrey.png",500),("pink.png",600),("black.png",700)]
CARS = [(pygame.image.load(assets.image(name)),price) for name,price in CAR_SKINS]
try:
    file = open(assets.PROGRESS_FILE,"rb")
    purchased,balance,selected = pickle.load(file)
except FileNotFoundError:
    purchased = [True,False,False,False,False,False,False,False]
    balance = 250
    selected = 0
NPC_START_POS = [[610,490],[700,540],[610,320],[610,150],[700,200]]
pygame.mixer.music.load(assets.audio("rockit.mp3"))
pygame.mixer.music.set_volume(0.7)
pygame.mixer.music.play(-1)
music_paused = False