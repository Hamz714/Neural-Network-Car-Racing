import pygame,math,pickle,sys,random
import window as w
import variable as v
import neural_network as n

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

# def improve_networks():
#     if v.end_generation.check():
#         store()
#         print(v.generation)
#         v.generation += 1
#         reset()
#         v.NPC_cars = n.best_networks(v.NPC_cars)
#         v.NPC_cars = n.mutation(v.NPC_cars,v.NPC)

# def reset():
#     v.player = v.PlayerCar()
#     v.track = v.Track()

# def store():
#     file_name = "generation" + str(v.generation) + ".txt"
#     file = open(file_name,"wb")
#     networks = []
#     for car in v.NPC_cars:
#         networks.append(car.network)
#     pickle.dump(networks,file)

def load(difficulty):
    NPC_cars = []
    file_name = difficulty + ".txt"
    file = open(file_name,"rb")
    network = pickle.load(file)
    colours_picked = [v.CARS[v.selected][0]]
    for i in range(5):
        colour = random.choice(v.CARS)[0]
        while colour in colours_picked:
            colour = random.choice(v.CARS)[0]
        colours_picked.append(colour)
        NPC_cars.append(v.NPC(colour,0,network))
    return NPC_cars

def update_progress():
    file = open("progress.txt","wb")
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