import pygame,math,pickle,sys,random
from copy import deepcopy
from nncar import window as w
from nncar import entities as v
from nncar import neural_network as n
from nncar import assets

def quit():
    if pygame.event.get(pygame.QUIT):
        sys.exit()

def update_game():
    w.window.fill((128,128,128))
    v.track.draw()
    draw_NPC()
    v.player.draw()
    draw_leaderboard()

def move():
    v.player.movement()
    for car in v.NPC_cars:
        car.update_sensors()
        car.move()

def draw_NPC():
    for car in v.NPC_cars:
        car.draw()

def NPC_collision():
    for i in range(len(v.NPC_cars)):
        if v.NPC_cars[i].collisions >= 5:
            v.NPC_cars[i] = v.NPC(v.NPC_cars[i].image,v.NPC_cars[i].laps,v.NPC_cars[i].network)
            v.track.leaderboard = v.NPC_cars + [v.player]

def checkpoints():
    v.player.reset_checkpoints()
    v.player.check_checkpoints()
    for car in v.NPC_cars:
        car.reset_checkpoints()
        car.check_checkpoints()

def update_leaderboard():
    v.track.leaderboard = merge_sort(v.track.leaderboard,key=custom_sort_key,reverse=True)
    for car in v.track.leaderboard:
        if car.type == "player":
            car.placement = v.track.leaderboard.index(car) + 1

def draw_leaderboard():
    y = 10
    for car in v.track.leaderboard:
        pygame.draw.rect(w.window,(0,0,0),(1350,y,35,20))
        transformed_image = pygame.transform.scale(car.image,(20,35))
        transformed_image = pygame.transform.rotate(transformed_image,90)
        w.window.blit(transformed_image,(1350,y))
        y += 30

def finish_race():
    for car in v.NPC_cars:
        if car.laps == v.track.lap_number:
            v.NPC_cars.remove(car)
    if v.player.laps == v.track.lap_number:
        return True

def load_model(difficulty):
    """Read a saved model, returning (networks, normalise_inputs).

    Accepts both the versioned dictionary written by the trainer and a bare
    pickled Network from before the format existed.
    """
    with open(assets.model(difficulty + ".pkl"),"rb") as file:
        payload = pickle.load(file)
    if isinstance(payload,dict):
        networks = payload.get("networks") or [payload["network"]]
        return networks,payload.get("normalise_inputs",True)
    return [payload],False


def load(difficulty):
    """Build the five opponent cars for a race.

    Each NPC gets its own network. The original code handed the same object to
    all five, so every opponent shared one brain and differed only by the noise
    added to its outputs each frame; when fewer champions are available than
    there are cars, they are cycled and copied so the cars stay independent.
    """
    networks,normalise = load_model(difficulty)
    NPC_cars = []
    colours_picked = [v.CARS[v.selected][0]]
    for i in range(5):
        colour = random.choice(v.CARS)[0]
        while colour in colours_picked:
            colour = random.choice(v.CARS)[0]
        colours_picked.append(colour)
        network = deepcopy(networks[i % len(networks)])
        car = v.NPC(colour,0,network)
        car.normalise_inputs = normalise
        NPC_cars.append(car)
    return NPC_cars

def update_progress():
    file = open(assets.PROGRESS_FILE,"wb")
    pickle.dump([v.purchased,v.balance,v.selected],file)

def calculate_balance(difficulty,lap_number):
    if difficulty == "easy":
        multiplier = 1
    elif difficulty == "medium":
        multiplier = 2
    else:
        multiplier = 3
    money_earned = (6-v.player.placement) * multiplier * lap_number
    v.balance += money_earned
    update_progress()
    return money_earned

def check_hovering(buttons):
    mouse_pos = pygame.mouse.get_pos()
    hover_effect = False
    for button in buttons:
        if button.hover(mouse_pos):
            hover_effect = True
            break
    if not hover_effect:
        pygame.mouse.set_visible(True)

def custom_sort_key(car):
    return car.laps,car.checkpoints_passed,-car.last_checkpoint

def merge_sort(cars, key=None, reverse=False):
    if len(cars) <= 1:
        return cars

    def merge(left, right):
        result = []
        i = 0
        j = 0
        while i < len(left) and j < len(right):
            if key:
                left_value = key(left[i])
                right_value = key(right[j])
            else:
                left_value = left[i]
                right_value = right[j]
            if reverse and left_value > right_value or not reverse and left_value < right_value:
                result.append(left[i])
                i += 1
            elif left_value == right_value:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    middle = len(cars) // 2
    left_half = cars[:middle]
    right_half = cars[middle:]
    left_sorted = merge_sort(left_half, key, reverse)
    right_sorted = merge_sort(right_half, key, reverse)
    return merge(left_sorted, right_sorted)