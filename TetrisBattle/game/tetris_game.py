import pygame
import argparse
from TetrisBattle.settings import *

import time as t

from TetrisBattle.renderer import Renderer

from TetrisBattle.tetris import Tetris, Player, Judge, collideDown, collide, collideLeft, collideRight, \
    hardDrop, freeze, get_infos

POS_LIST = [
    {
        'combo': (44, 437),
        'tetris': (314, 477),
        'tspin': (304, 477),
        'back2back': (314, 437),
        'board': (112, 138),
        'drawscreen': (112, 138),
        'big_ko': (44, 235),
        'ko': (140, 233),
        'transparent': (110, 135),
        'gamescreen': (0, 0), 
        'attack_clean': (298, 140, 3, 360),
        'attack_alarm': (298, 481, 3, 18),
        'you_win': (120, 230),
        'you_loss': (515, 230),
    },
    {
        'combo': (415, 437),
        'tetris': (685, 477),
        'tspin': (675, 477),
        'back2back': (685, 437),
        'board': (495, 138),
        'drawscreen': (495, 138),
        'big_ko': (426, 235),
        'ko': (527, 233),
        'transparent': (494, 135),
        'gamescreen': (0, 0), 
        'attack_clean': (680, 140, 3, 360),
        'attack_alarm': (680, 481, 3, 18),
        'you_win': (520, 230),
        'you_loss': (115, 230),
    }
]

class TetrisGame:
    #first function
    #will be used for choosing what map you want to play on

    def __init__(self):
        import os 
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        # SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT), pygame.FULLSCREEN) # SCREEN is 800*600 
        self.screen = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT)) # SCREEN is 800*600 
        images = load_imgs()
        self.renderer = Renderer(self.screen, images)

    def play(self):
        page = "menu"
        myClock = pygame.time.Clock() # this will be used to set the FPS(frames/s) 

        timer2p = pygame.time.Clock() # this will be used for counting down time in our game
        while page != "exit":
            if page == "menu":
                page = self.menu(page)
            if page == "start":
                page = self.start(myClock, timer2p)
            if page == "viewmap":
                page = self.viewmap()    
            if page == "instructions":
                page = self.instructions()    
            # if page == "pygame.quit":
            #     page = (page)    
            # if page == "credits":
            #     page = self.credit(page)
        # pygame.quit()

    def setmap(self):
        running = True
        # loading images
        
        # defining rectangles for collision checking
        map1 = pygame.Rect(155, 75, 200, 200)
        map2 = pygame.Rect(439, 75, 200, 200)
        map3 = pygame.Rect(155, 309, 200, 200)
        map4 = pygame.Rect(439, 309, 200, 200)
        buttons = [map1, map2, map3, map4]# preparing a buttons and names list to be zipped together
        names = ["none", "classic", "comboking", "lunchbox"]# 
        self.renderer.drawByName("back1", 0, 0) # IMAGES["back1"] is the main background
        while running:
            mpos = pygame.mouse.get_pos()
            mb = pygame.mouse.get_pressed()
            for evnt in pygame.event.get():          
                if evnt.type == pygame.QUIT:
                    running = False

            for b, n in zip(buttons, names): # zipping buttons and names together       
                if b.collidepoint(mpos):   # for very easy collision checking            
                    if mb[0] == 1:
                        return n # return the name of the map chosen

            # this chunk of code is just making pretty pictures       
            if map1.collidepoint(mpos):
                self.renderer.drawByName("outline", 149, 64)
            elif map2.collidepoint(mpos):
                self.renderer.drawByName("outline", 431, 64)
            elif map3.collidepoint(mpos):
                self.renderer.drawByName("outline", 149, 301)
            elif map4.collidepoint(mpos):
                self.renderer.drawByName("outline", 149, 301)
            else:
                self.renderer.drawByName("back1", 0, 0) # keeping it fresh
            
            pygame.display.flip() #necessities
    #gridchoice=""    

    #################################################################################
    def viewmap(self):#viewmap function
        running = True
        #making it look nice
    #defining rectangles for collision checking
        map1 = pygame.Rect(155, 75, 200, 200)
        map2 = pygame.Rect(439, 75, 200, 200)
        map3 = pygame.Rect(155, 309, 200, 200)
        map4 = pygame.Rect(439, 309, 200, 200)
        buttons = [map1, map2, map3, map4]#preparing a buttons and names list to be zipped together
        names = ["none", "classic", "comboking", "lunchbox"]# 
        self.renderer.drawByName("back1", 0, 0) # back1 is the main background
        while running:

            for evnt in pygame.event.get():          
                if evnt.type == pygame.QUIT:
                    running = False

            mpos = pygame.mouse.get_pos()
            mb = pygame.mouse.get_pressed()
            #print mpos

            #for b,n in zip(buttons,names): #zipping buttons and names together       
                #if b.collidepoint(mpos):   #for very easy collision checking            
                    #if mb[0]==1:
                        

            #this chunk of code is just making pretty pictures       
            if map1.collidepoint(mpos):
                self.renderer.drawByName("outline", 149, 64)
            elif map2.collidepoint(mpos):
                self.renderer.drawByName("outline", 431, 64)
            elif map3.collidepoint(mpos):
                self.renderer.drawByName("outline", 149, 301)
            elif map4.collidepoint(mpos):
                self.renderer.drawByName("outline", 149, 301)
            else:
                self.renderer.drawByName("back1", 0, 0) # keeping it fresh

            pygame.display.flip() #necessities

        return "menu"

    #instructions page
    def instructions(self):
        running = True
        self.renderer.drawByName("back2", 0, 0)
        #SCREEN.blit(inst,(173,100))
        while running:
            for evnt in pygame.event.get():          
                if evnt.type == pygame.QUIT:
                    running = False
            pygame.display.flip()

        return "menu"

    #menu page
    def menu(self, page):
        running = True
        myClock = pygame.time.Clock()
        button1 = pygame.Rect(320, 204, 146, 50)#start rect
        buttons = [pygame.Rect(325, y * 42 + 275, 135, 30) for y in range(3)]#other three rects 
        vals = ["viewmap", "instructions", "exit"]#values of other three rects
        
        
        self.renderer.drawByName("intro", 0, 0)
        pygame.display.set_caption("Tetris Battle", "tetris battle")
        while running:
            for evnt in pygame.event.get():
                if evnt.type == pygame.QUIT:
                    return "exit"
            mpos = pygame.mouse.get_pos()
            mb = pygame.mouse.get_pressed()
            #print mpos

            #thank u for the code
            #zips button and vals and when
            #a collision and a click occurs
            #returns vals
            for r,v in zip(buttons, vals):
                if r.collidepoint(mpos):
                    #print r,v
                    if mb[0] == 1:
                        return v # page to go to
           
            if button1.collidepoint(mpos):
                self.renderer.drawByName("startbutton", 319, 207)
                if mb[0] == 1:
                    return "start"#starting the game
                #making the game look pretty
            elif buttons[0].collidepoint(mpos):
                self.renderer.drawByName("startbutton", 319, 207)
                self.renderer.drawByName("setmapbutton", 325, 274)
            elif buttons[1].collidepoint(mpos):
                self.renderer.drawByName("helpbutton", 325, 317)
            elif buttons[2].collidepoint(mpos):
                self.renderer.drawByName("quitbutton", 325, 360)
                if mb[0] == 1:
                    return "exit" # quitting the game
            else:
                self.renderer.drawByName("intro", 0, 0) # reblit the background
            
            #draw.rect(SCREEN,(0,0,0),button1,2)
            myClock.tick(FPS)            
            pygame.display.flip()

    #main game function
    def start(self, myClock, timer2p):
        raise NotImplementedError
    

class TetrisGameDouble(TetrisGame):

    def __init__(self):
        super(TetrisGameDouble, self).__init__()
        self.num_players = 2

    def start(self, myClock, timer2p):#parameters are FP/s rate and timer countdown
    ################################################################################
        
        gridchoice = self.setmap()#calling the setmap function for a choice of grid
        timer2p.tick()
        #the code below is what happens when you set a map
        #different maps = differnet grids
               
        pygame.init() #for music
        battlemusic = pygame.mixer.Sound(MUSIC_PATH)#importing sound file

        #SCREEN=pygame.display.set_mode((800,600))
        running = True #necessity
        # SCREEN.blit(IMAGES["gamescreen"], (0, 0))# blitting the main background
        self.renderer.drawByName("gamescreen", 0, 0)

        #these two used for countdown
        #of the timer
        time = MAX_TIME
        delaytime = time

        info_dict_list = [
            {
            "id": 0,
            "hold": pygame.K_c,
            "drop": pygame.K_SPACE,
            "rotate_right": pygame.K_UP,
            "rotate_left": pygame.K_z,
            "right": pygame.K_RIGHT,
            "left": pygame.K_LEFT,
            "down": pygame.K_DOWN
            },
            {
            "id": 1,
            "hold": pygame.K_e,
            "drop": pygame.K_w,
            "rotate_right": pygame.K_u,
            "rotate_left": pygame.K_q,
            "right": pygame.K_k,
            "left": pygame.K_h,
            "down": pygame.K_j
            }
        ]

        tetris_list = []

        for i in range(self.num_players):
            tetris_list.append({
                'info_dict': info_dict_list[i],
                'tetris': Tetris(Player(info_dict_list[i]), gridchoice),
                'pos': POS_LIST[i]
            })

        winner = 0
        force_quit = 0
        #main loop
        while running:
            # battlemusic.play()#plays music
            
            for tetris_dict in tetris_list:
                tetris_dict["tetris"].natural_down()

            for evt in pygame.event.get():
                if evt.type == pygame.QUIT:
                    running = False
                    force_quit = 1

                for tetris_dict in tetris_list:
                    tetris_dict["tetris"].trigger(evt)

            for tetris_dict in tetris_list:
                tetris_dict["tetris"].move()

            for i, tetris_dict in enumerate(tetris_list):
                opponent = tetris_list[self.num_players - 1 - i]
                tetris, pos = tetris_dict["tetris"], tetris_dict["pos"]

                if tetris.check_fallen():
                    # compute the scores and attack the opponent
                    scores = tetris.clear()

                    opponent["tetris"].add_attacked(scores)

                    self.renderer.drawCombo(tetris, *pos["combo"])

                    self.renderer.drawTetris(tetris, *pos["tetris"])
                    self.renderer.drawTspin(tetris, *pos["tspin"])
                    self.renderer.drawBack2Back(tetris, *pos["back2back"])

                    if tetris.check_KO():
                        
                        self.renderer.drawBoard(tetris, *pos["board"])
                        
                        opponent["tetris"].update_ko()

                        tetris.clear_garbage()

                        self.renderer.drawByName("ko", *pos["ko"])
                        self.renderer.drawByName("transparent", *pos["transparent"])

                        # screen.blit(kos[tetris_2.get_KO() - 1], (426, 235))
                        pygame.display.flip()
                        freeze(0.5)
                        # scores -= 1

                        # end = 1

                    tetris.new_block()

                self.renderer.drawGameScreen(tetris)

                if tetris.attacked == 0:
                    pygame.draw.rect(self.screen, (30, 30, 30), pos["attack_clean"]) 

                if tetris.attacked != 0:
                    
                    for j in range(tetris.attacked):
                        pos_attack_alarm = list(pos["attack_alarm"])
                        # modified the y axis of the rectangle, according to the strength of attack
                        pos_attack_alarm[1] = pos_attack_alarm[1] - 18 * j
                        pygame.draw.rect(self.screen, (255, 0, 0), pos_attack_alarm) 

                if tetris.KO > 0:
                    self.renderer.drawKO(tetris.KO, *pos["big_ko"])
                    
                self.renderer.drawScreen(tetris, *pos["drawscreen"])

                if Judge.check_ko_win(tetris, max_ko=3):
                    running = False
                    winner = tetris.get_id()

                if Judge.check_ko_win(opponent["tetris"], max_ko=3):
                    running = False
                    winner = opponent["tetris"].get_id()

                if time >= 0:
                    self.time -= timer2p.tick() * SPEED_UP
                else:
                    time = 0
                    winner = Judge.who_win(tetris, opponent["tetris"])

                    running = False

            self.renderer.drawTime2p(time)

            myClock.tick(FPS)   
            pygame.display.flip()

        if force_quit:
            return "menu"


        for i, tetris_dict in enumerate(tetris_list):
            opponent = tetris_list[self.num_players - 1 - i]
            tetris, pos = tetris_dict["tetris"], tetris_dict["pos"]
            self.renderer.drawByName("transparent", *pos["transparent"])
            if i == winner: # is winner
                self.renderer.drawByName("you_win", *pos["you_win"])
            else:
                self.renderer.drawByName("you_lose", *pos["you_lose"])
        # # end game
        # # self.renderer.drawByObj(a, 0, 0)
        # self.renderer.drawByName("transparent", 110, 135)
        # self.renderer.drawByName("transparent", 494, 135)

        # if winner == 0:
        #     self.renderer.drawByName("you_win", 120, 230)
        #     self.renderer.drawByName("you_lose", 515, 230)
        # else:
        #     self.renderer.drawByName("you_win", 520, 230)
        #     self.renderer.drawByName("you_lose", 115, 230)

        pygame.display.flip()

        freeze(2.0)

        return "menu"

        # pygame.quit()

class TetrisGameSingle(TetrisGame):

    def __init__(self):
        super(TetrisGameSingle, self).__init__()
        self.num_players = 1

    def start(self, myClock, timer2p):#parameters are FP/s rate and timer countdown
    ################################################################################

        gridchoice = self.setmap()#calling the setmap function for a choice of grid

        timer2p.tick()
        #the code below is what happens when you set a map
        #different maps = differnet grids

        pygame.init() #for music

        battlemusic = pygame.mixer.Sound(MUSIC_PATH)#importing sound file
        #SCREEN=pygame.display.set_mode((800,600))
        running = True #necessity
        self.renderer.drawByName("gamescreen", 0, 0)
        # a = SCREEN.copy()#used for image coverage

        #these two used for countdown
        #of the timer
        time = MAX_TIME
        delaytime = time

        info_dict = {
        "id": 0,
        "hold": pygame.K_c,
        "drop": pygame.K_SPACE,
        "rotate_right": pygame.K_UP,
        "rotate_left": pygame.K_z,
        "right": pygame.K_RIGHT,
        "left": pygame.K_LEFT,
        "down": pygame.K_DOWN
        }

        tetris = Tetris(Player(info_dict), gridchoice)

        pos = POS_LIST[info_dict["id"]]

        opponent_pos = POS_LIST[-1]
        # print("213")
        
        #main loop
        force_quit = 0
        kk = 0
        while running:
            # battlemusic.play()#plays music
            
            tetris.natural_down()

            for evt in pygame.event.get():
                if evt.type == pygame.QUIT:
                    running = False
                    force_quit = 1
                tetris.trigger(evt)

            tetris.move()

            if tetris.check_fallen():
                # compute the scores and attack the opponent
                scores = tetris.clear()

                self.renderer.drawCombo(tetris, *pos["combo"])

                self.renderer.drawTetris(tetris, *pos["tetris"])
                self.renderer.drawTspin(tetris, *pos["tspin"])
                self.renderer.drawBack2Back(tetris, *pos["back2back"])

                if tetris.check_KO():
                    
                    self.renderer.drawBoard(tetris, *pos["board"])
                    
                    tetris.clear_garbage()

                    pygame.display.flip()
                    running = False

                tetris.new_block()

            self.renderer.drawGameScreen(tetris)

            self.renderer.drawScreen(tetris, *pos["drawscreen"])

            self.renderer.drawByName("transparent", *opponent_pos["transparent"])

            # pygame.display.update(r)

            if time >= 0:
                time -= timer2p.tick() * SPEED_UP
            else:
                time = 0
                running = False

            self.renderer.drawTime2p(time)

            # time goes until it hits zero
            # when it hits zero return endgame SCREEN
            myClock.tick(FPS)   
            pygame.display.flip()

        if force_quit:
            return "menu"
        # self.renderer.drawByObj(a, 0, 0)
        self.renderer.drawByName("transparent", *pos["transparent"])

        pygame.display.flip()
        freeze(2.0)
        # pygame.quit()
        return "menu"

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "double"], default="single")

    return parser.parse_args()

if __name__ == "__main__":
    args = parser()
    if args.mode == "single":
        game = TetrisGameSingle()
    else:
        game = TetrisGameDouble()
    game.play()