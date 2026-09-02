import pygame,time,sys
from copy import deepcopy
from nncar import game as f
from nncar import entities as v
from nncar import window as w
from nncar import assets
pygame.init()
pygame.mixer.init()

fps = 50
clock = pygame.time.Clock()

def main():
    font = pygame.font.Font(None,150)
    title = font.render("CAR RACING",True,(0,0,0))
    start_button = v.Action_Button(550,250,300,100,"start",start)
    shop_button = v.Action_Button(550,370,300,100,"shop",shop)
    howtoplay_button = v.Action_Button(550,490,300,100,"howtoplay",howtoplay)
    quit_button = v.Action_Button(550,610,300,100,"quit",quit)
    action_buttons = [start_button,shop_button,howtoplay_button,quit_button]
    pause_music_button = v.Option_Button(50,50,100,100,"pausemusic",True)
    unpause_music_button = v.Option_Button(50,50,100,100,"unpausemusic",True)
    buttons = action_buttons + [pause_music_button,unpause_music_button]

    while True:
        f.quit()
        mouse_pressed = pygame.event.get(pygame.MOUSEBUTTONDOWN)
        mouse_pos = pygame.mouse.get_pos()
        w.window.fill((100,100,100))
        w.window.blit(title,(350,70))
        for button in action_buttons:
            button.update()
        if v.music_paused:
            pygame.mixer.music.pause()
            unpause_music_button.update()
            if mouse_pressed:
                if unpause_music_button.pressed(mouse_pos):
                    v.music_paused = not v.music_paused
        else:
            pygame.mixer.music.unpause()
            pause_music_button.update()
            if mouse_pressed:
                if pause_music_button.pressed(mouse_pos):
                    v.music_paused = not v.music_paused
        if mouse_pressed:
            for button in action_buttons:
                button.pressed(mouse_pos)
        f.check_hovering(buttons)
        pygame.display.flip()

def start():
    font = pygame.font.Font(None,100)
    global difficulty
    difficulty = "easy"
    easy_button = v.Option_Button(200,200,200,100,"easy","easy")
    medium_button = v.Option_Button(500,200,200,100,"medium","medium")
    hard_button = v.Option_Button(800,200,200,100,"hard","hard")
    difficulty_buttons = [easy_button,medium_button,hard_button]
    global lap_number
    lap_number = 1
    one_lap = v.Option_Button(200,480,150,100,"1",1)
    two_lap = v.Option_Button(400,480,150,100,"2",2)
    three_lap = v.Option_Button(600,480,150,100,"3",3)
    four_lap = v.Option_Button(800,480,150,100,"4",4)
    five_lap = v.Option_Button(1000,480,150,100,"5",5)
    lap_buttons = [one_lap,two_lap,three_lap,four_lap,five_lap]
    game_button = v.Action_Button(600,630,200,100,"start",game)
    return_button = v.Action_Button(50,50,100,100,"return",main)
    buttons = difficulty_buttons + lap_buttons + [game_button,return_button]

    while True:
        f.quit()
        w.window.fill((100,100,100))
        for button in buttons:
            button.update()
        if pygame.event.get(pygame.MOUSEBUTTONDOWN):
            mouse_pos = pygame.mouse.get_pos()
            game_button.pressed(mouse_pos)
            return_button.pressed(mouse_pos)
            for button in difficulty_buttons:
                new_difficulty = button.pressed(mouse_pos)
                if new_difficulty != None:
                    difficulty = new_difficulty
                    break
            for button in lap_buttons:
                new_lap_number = button.pressed(mouse_pos)
                if new_lap_number != None:
                    lap_number = new_lap_number
                    break
        difficulty_text = font.render("Difficulty: " + difficulty, True, (255,255,255))
        w.window.blit(difficulty_text,(200,70))
        lap_number_text = font.render("Lap Number: " + str(lap_number), True, (255,255,255))
        w.window.blit(lap_number_text,(200,350))
        f.check_hovering(buttons)
        pygame.display.flip()

def game():
    font = pygame.font.Font(None,50)
    v.player = v.PlayerCar()
    v.track = v.Track(lap_number)
    v.NPC.start_positions = deepcopy(v.NPC_START_POS)
    v.NPC_cars = f.load(difficulty)
    v.track.leaderboard = v.NPC_cars + [v.player]
    leave_button = v.Action_Button(600,370,200,100,"leave",main)
    pause_button = v.Option_Button(10,10,50,50,"pause",True)
    resume_button = v.Option_Button(600,250,200,100,"resume",False)
    paused = False
    escape_key_pressed = False
    pause_music_button = v.Option_Button(70,10,50,50,"pausemusic",True)
    unpause_music_button = v.Option_Button(70,10,50,50,"unpausemusic",True)
    buttons1 = [leave_button,resume_button]
    buttons2 = [pause_button,pause_music_button,unpause_music_button]

    while True:
        mouse_pressed = pygame.event.get(pygame.MOUSEBUTTONDOWN)
        mouse_pos = pygame.mouse.get_pos()
        if paused:
            f.quit()
            pygame.draw.rect(w.window,(100,100,100),(500,75,400,600))
            for button in buttons1:
                button.update()
            f.check_hovering(buttons1)
            if mouse_pressed:
                leave_button.pressed(mouse_pos)
                new_paused = resume_button.pressed(mouse_pos)
                if new_paused != None:
                    paused = new_paused
            elif pygame.key.get_pressed()[pygame.K_ESCAPE] and not escape_key_pressed:
                paused = not paused
                escape_key_pressed = True
            elif not pygame.key.get_pressed()[pygame.K_ESCAPE]:
                escape_key_pressed = False
            pygame.display.flip()
        else:
            f.quit()
            if mouse_pressed:
                paused = pause_button.pressed(mouse_pos)
            elif pygame.key.get_pressed()[pygame.K_ESCAPE] and not escape_key_pressed:
                paused = not paused
                escape_key_pressed = True
            elif not pygame.key.get_pressed()[pygame.K_ESCAPE]:
                escape_key_pressed = False
            f.move()
            f.checkpoints()
            f.NPC_collision()
            f.update_leaderboard()
            f.update_game()
            pause_button.update()
            if v.music_paused:
                pygame.mixer.music.pause()
                unpause_music_button.update()
                if mouse_pressed:
                    if unpause_music_button.pressed(mouse_pos):
                        v.music_paused = not v.music_paused
            else:
                pygame.mixer.music.unpause()
                pause_music_button.update()
                if mouse_pressed:
                    if pause_music_button.pressed(mouse_pos):
                        v.music_paused = not v.music_paused
            if v.player.placement == 1:
                ordinal = "st"
            elif v.player.placement == 2:
                ordinal = "nd"
            elif v.player.placement == 3:
                ordinal = "rd"
            else:
                ordinal = "th" 
            placement_text = font.render(str(v.player.placement)+ordinal+" place",True,(255,255,255))
            w.window.blit(placement_text,(1230,650))
            laps_text = font.render("Lap: "+str(v.player.laps)+"/"+str(v.track.lap_number),True,(255,255,255))
            w.window.blit(laps_text,(1250,700))
            if f.finish_race():
                finish(ordinal)
            f.check_hovering(buttons2)
            pygame.display.flip()
            clock.tick(fps)

def finish(ordinal):
    font = pygame.font.Font(None,50)
    placement_text = font.render(str(v.player.placement)+ordinal+" place",True,(255,255,255))
    money_earned = f.calculate_balance(difficulty,lap_number)
    money_earned = font.render("Money Earned: $"+str(money_earned),True,(255,255,255))
    finish_button = v.Action_Button(600,500,200,100,"finish",main)

    while True:
        f.quit()
        pygame.draw.rect(w.window,(100,100,100),(500,75,400,600))
        w.window.blit(placement_text,(620,200))
        w.window.blit(money_earned,(550,300))
        finish_button.update()
        if pygame.event.get(pygame.MOUSEBUTTONDOWN):
            mouse_pos = pygame.mouse.get_pos()
            finish_button.pressed(mouse_pos)
        f.check_hovering([finish_button])
        pygame.display.flip()

def shop():
    font = pygame.font.Font(None,50)
    return_button = v.Action_Button(50,50,100,100,"return",main)
    purchase_button = v.Option_Button(600,500,200,100,"purchase",True)
    select_button = v.Option_Button(600,500,200,100,"select",True)
    selected_button = v.Option_Button(600,500,200,100,"selected",True)
    previous_button = v.Option_Button(50,300,100,100,"previous",True)
    next_button = v.Option_Button(1250,300,100,100,"next",True)
    buttons = [return_button,purchase_button,select_button,selected_button,previous_button,next_button]
    display_pos = 0

    while True:
        f.quit()
        mouse_pressed = pygame.event.get(pygame.MOUSEBUTTONDOWN)
        mouse_pos = pygame.mouse.get_pos()
        w.window.fill((100,100,100))
        pygame.draw.rect(w.window,(100,100,100),(500,75,400,600))
        balance = font.render("Balance: $"+str(v.balance),True,(255,255,255))
        w.window.blit(balance,(1000,50))
        return_button.update()
        previous_button.update()
        next_button.update()
        if mouse_pressed:
            return_button.pressed(mouse_pos)
            if previous_button.pressed(mouse_pos):
                if display_pos > 0:
                    display_pos -= 1
            elif next_button.pressed(mouse_pos):
                if display_pos < 7:
                    display_pos += 1
        if v.purchased[display_pos]:
            if v.selected == display_pos:
                selected_button.update()
            else:
                select_button.update()
                if mouse_pressed:
                    if select_button.pressed(mouse_pos):
                        v.selected = display_pos
                        f.update_progress()
        else:
            purchase_button.update()
            price = font.render("Price: $"+str(v.CARS[display_pos][1]),True,(255,255,255))
            w.window.blit(price,(610,460))
            if mouse_pressed:
                if purchase_button.pressed(mouse_pos) and v.balance >= v.CARS[display_pos][1]:
                    v.purchased[display_pos] = True
                    v.balance -= v.CARS[display_pos][1]
                    f.update_progress()
        displayed_car = pygame.transform.scale(v.CARS[display_pos][0],(200,350))
        w.window.blit(displayed_car,(600,100))
        f.check_hovering(buttons)
        pygame.display.flip()

def howtoplay():
    text = pygame.image.load(assets.image("howtoplay.png"))
    return_button = v.Action_Button(30,20,100,100,"return",main)

    while True:
        f.quit()
        w.window.fill((100,100,100))
        w.window.blit(text,(0,0))
        return_button.update()
        if pygame.event.get(pygame.MOUSEBUTTONDOWN):
            mouse_pos = pygame.mouse.get_pos()
            return_button.pressed(mouse_pos)
        f.check_hovering([return_button])
        pygame.display.flip()

def quit():
    sys.exit()

if __name__ == "__main__":
    main()