#!/usr/bin/env python
#Q-learning
# 2016.11.06
#1.0  加入初始先验信息的强化学习路径规划   经过障碍物的试错学习
import sys, random, math, pygame
from pygame.locals import *
from math import sqrt,cos,sin,atan2
import numpy as np
#from RRT_includes import *

class Node(object):
    """Node in a tree"""
    def __init__(self, point, parent):
        super(Node, self).__init__()
        self.point = point
        self.parent = parent


def dist(p1,p2):
    return sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))

def point_circle_collision(p1, p2, radius):
    distance = dist(p1,p2)
    if (distance <= radius):
        return True
    return False
    
#constants
XDIM = 800
YDIM = 500
WINSIZE = [XDIM, YDIM]
EPSILON = 7.0
NUMNODES = 10000
GOAL_RADIUS = 10
MIN_DISTANCE_TO_ADD = 1.0
GAME_LEVEL = 1

pygame.init()
fpsClock = pygame.time.Clock()

#initialize and prepare screen
screen = pygame.display.set_mode(WINSIZE)
pygame.display.set_caption('Q-learning')
white = 255, 240, 200
black = 20, 20, 40
red = 255, 0, 0
blue = 0, 255, 0
green = 0, 0, 255
cyan = 0,255,255


#强化学习相应参数设置
#eps=0.2    #贪心法则
gamma=0.9      #学习率
loss=0.99    #损失率
epochs=2500   #迭代次数
Q_table=np.zeros((102,162))  #50*80的二维数组
#Q_table=np.where(Q_table==0,-100.0,-100.0)
Pmin=10      #最小前进单位

#初始化Q值，距离目标越近，Q值越大，障碍物Q值为0
def init_Q_table(Current_Point,Goal_point_table):
    global Q_table
    for i in range(0,805,5):
       for j in range(0,505,5):  
          if collides((i,j)) == False:
             Goal_point_table[0]=(Goal_point_table[0]//5)*5
             Goal_point_table[1]=(Goal_point_table[1]//5)*5
             if(dist(Current_Point,Goal_point_table)>=dist([i,j],Goal_point_table)):
               Q=500/(dist([i,j],Goal_point_table)+10)      #初始化Q值
               x=j//5
               y=i//5
               Q_table[x][y]+=Q
    return 0

#设置回报值
def reward(Current_Point,goal_point):
    if(dist(Current_Point,goal_point)<=5):
        r=500
        return r
    elif(collides(Current_Point) == True):
        r=-2000
        return r
    else:
        r=0
        return r
    
     
#更新Q_table    
def update_Q_table(q_Current_Point,reward):
    global Q_table
    x=q_Current_Point[0]//5   #转换为换算单位
    y=q_Current_Point[1]//5
    Q_table[y][x]=(1-gamma)*(Q_table[y][x])+gamma*(reward+loss*max_Q_nextvalue(q_Current_Point))
    #print("Q",x,y,Q_table[y][x])
    
def updata_reward_Q_table(Current_Point,reward):   #目标点Q值为reward值
    global Q_table
    x=Current_Point[0]//5   #转换为换算单位
    y=Current_Point[1]//5
    Q_table[y][x]=reward
#预测下一步最大的Q值    
def max_Q_nextvalue(Current_Point):
    x=Current_Point[0]//5
    y=Current_Point[1]//5    
    if((x<=0)and(y<=0)):
         max_Q=np.max([Q_table[y][x+1],Q_table[y+1][x],Q_table[y+1][x+1]])
         return max_Q      
    elif((x>=160)and(y<=0)):
         max_Q=np.max([Q_table[y][x-1],Q_table[y+1][x],Q_table[y+1][x-1]])
         return max_Q  
    elif((x<=0)and(y>=99)):
         max_Q=np.max([Q_table[y-1][x],Q_table[y][x+1],Q_table[y-1][x+1]])
         return max_Q   
    elif((x>=160)and(y>=99)):
         max_Q=np.max([Q_table[y][x-1],Q_table[y-1][x],Q_table[y-1][x-1]])
         return max_Q  
    elif(x<=0):
         max_Q=np.max([Q_table[y-1][x],Q_table[y][x+1],Q_table[y+1][x],Q_table[y+1][x+1],Q_table[y-1][x+1]])
         return max_Q  
    elif(y<=0):
         max_Q=np.max([Q_table[y][x-1],Q_table[y][x+1],Q_table[y+1][x],Q_table[y+1][x+1],Q_table[y+1][x-1]])
         return max_Q  
    elif(x>=160):
         max_Q=np.max([Q_table[y][x-1],Q_table[y-1][x],Q_table[y+1][x],Q_table[y+1][x-1],Q_table[y-1][x-1]])
         return max_Q  
    elif(y>=99):
         max_Q=np.max([Q_table[y][x-1],Q_table[y-1][x],Q_table[y][x+1],Q_table[y-1][x+1],Q_table[y-1][x-1]])
         return max_Q  
    else:
         max_Q=np.max([Q_table[y][x-1],Q_table[y-1][x],Q_table[y][x+1],Q_table[y+1][x],Q_table[y+1][x+1],Q_table[y-1][x+1],Q_table[y+1][x-1],Q_table[y-1][x-1]])
         return max_Q   
    
#贪心法则,下一步动作
def max_Q_action(Current_Point):
    x=Current_Point[0]//5  #整除 获得最小分辨单位
    y=Current_Point[1]//5
    if(x<=1):
        x=1
    if(y<=1):
        y=1
    if(x>=160):
        x=160
    if(y>=99):
        y=99
    greedy_action=np.argmax([Q_table[y][x-1],Q_table[y-1][x],Q_table[y][x+1],Q_table[y+1][x],Q_table[y+1][x+1],Q_table[y-1][x+1],Q_table[y+1][x-1],Q_table[y-1][x-1]])
    #print(greedy_action,Q_table[y][x-1],Q_table[y-1][x],Q_table[y][x+1],Q_table[y+1][x],Q_table[y+1][x+1],Q_table[y-1][x+1],Q_table[y+1][x-1],Q_table[y-1][x-1])
    if(greedy_action==0):
        return [5*(x-1),5*y]
    elif(greedy_action==1):
        return [5*x,5*(y-1)]
    elif(greedy_action==2):
        return [5*(x+1),5*y]
    elif(greedy_action==3):
        return [5*x,5*(y+1)]
    elif(greedy_action==4):
        return [5*(x+1),5*(y+1)]
    elif(greedy_action==5):
        return [5*(x+1),5*(y-1)]
    elif(greedy_action==6):
        return [5*(x-1),5*(y+1)]
    elif(greedy_action==7):
        return [5*(x-1),5*(y-1)]

#随机动作， 
def random_action(Current_Point) :
    x=Current_Point[0]//5
    y=Current_Point[1]//5
    if((y<=0)and(x<=0)):
        return[Current_Point, [5*(x+1),5*(y+1)]]
    elif((x<=0)and(y>=99)):
        return [Current_Point,[5*(x+1),5*(y-1)]]
    elif((x>=160)and(y<=0)):
        return [Current_Point,[5*(x-1),5*(y+1)]]
    elif((x>=160)and(y>=99)):
        return [Current_Point,[5*(x-1),5*(y-1)]]
    elif(y<=0):
        random_value=np.random.randint(4,9)
        if(random_value==4):
            return [Current_Point,[5*(x-1),5*y]]
        elif(random_value==5):
            return [Current_Point,[5*(x+1),5*y]]
        elif( 6<=random_value<=8):
            return [Current_Point,[5*(x-7+random_value),5*(y+1)]]
    elif(y>=99):
        random_value=np.random.randint(1,6)
        if(1<= random_value<=3):
            return [Current_Point,[5*(x-2+random_value),5*(y-1)]]
        elif(random_value==4):
            return [Current_Point,[5*(x-1),5*y]]
        elif(random_value==5):
            return [Current_Point,[5*(x+1),5*y]]
    elif(x<=0):
        random_value=np.random.randint(4,9)
        if(random_value==4):
            return [Current_Point,[5*x,5*(y-1)]]
        elif(random_value==5):
            return [Current_Point,[5*x,5*(y+1)]]
        elif( 6<=random_value<=8):
            return [Current_Point,[5*(x+1),5*(x-7+random_value)]]
    elif(x>=160):
        random_value=np.random.randint(1,6)
        if(1<= random_value<=3):
            return [Current_Point,[5*(x-1),5*(y-2+random_value)]]
        elif(random_value==4):
            return [Current_Point,[5*x,5*(y-1)]]
        elif(random_value==5):
            return [Current_Point,[5*x,5*(y+1)] ]
    else:
        random_value=np.random.randint(1,9)
        if(1<= random_value<=3):
            return [Current_Point,[5*(x-2+random_value),5*(y-1)]]
        elif(random_value==4):
            return [Current_Point,[5*(x-1),5*y]]
        elif(random_value==5):
            return [Current_Point,[5*(x+1),5*y]]
        elif( 6<=random_value<=8):
            return [Current_Point,[5*(x-7+random_value),5*(y+1)]]
  
    
def judge_goalfound(Current_Point,Goal_point):
    if(dist(Current_Point,Goal_point)<=6):
        return True
    return False
    

# setup program variables
count = 0
rectObs = []

#def dist(p1,p2):
  #  return sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))

def step_from_to(p1,p2):
    if dist(p1,p2) < EPSILON:
        return p2
    else:
        theta = atan2(p2[1]-p1[1],p2[0]-p1[0])
        return p1[0] + EPSILON*cos(theta), p1[1] + EPSILON*sin(theta)

def collides(p):
    for rect in rectObs:
        if rect.collidepoint(p) == True:
            # print ("collision with object: " + str(rect))
            return True
    return False


def get_random():
    return random.random()*XDIM, random.random()*YDIM

def get_random_clear():
    while True:
        p = get_random()
        noCollision = collides(p)
        if noCollision == False:
            return p


def init_obstacles(configNum):
    global rectObs
    rectObs = []
    #print("config "+ str(configNum))
    if (configNum == 0):
        rectObs.append(pygame.Rect((XDIM / 2.0 - 50, YDIM / 2.0 - 100),(100,200)))
    if (configNum == 1):
        rectObs.append(pygame.Rect((40,20),(20,350)))
        rectObs.append(pygame.Rect((120,180),(20,300)))
        rectObs.append(pygame.Rect((60,100),(80,20)))
        rectObs.append(pygame.Rect((140,0),(20,120)))
        rectObs.append(pygame.Rect((140,300),(80,20)))
        rectObs.append(pygame.Rect((200,400),(150,20)))
        rectObs.append(pygame.Rect((280,200),(20,200)))
        rectObs.append(pygame.Rect((300,480),(250,20)))
        rectObs.append(pygame.Rect((350,0),(20,300))) 
        rectObs.append(pygame.Rect((350,400),(20,100))) 
        rectObs.append(pygame.Rect((400,340),(100,20)))
        rectObs.append(pygame.Rect((450,200),(150,20)))
        rectObs.append(pygame.Rect((550,0),(20,150)))
        rectObs.append(pygame.Rect((550,250),(20,500))) 
        rectObs.append(pygame.Rect((620,50),(80,20))) 
        rectObs.append(pygame.Rect((620,300),(80,20)))
        rectObs.append(pygame.Rect((700,50),(20,270)))
    if (configNum == 2):
        rectObs.append(pygame.Rect((40,10),(100,200)))
    if (configNum == 3):
        rectObs.append(pygame.Rect((40,10),(100,200)))

    for rect in rectObs:
        pygame.draw.rect(screen, red, rect)


def reset():
  
    screen.fill(black)
    init_obstacles(GAME_LEVEL)
    


def main():
    global count
    step=0
    initPoseSet = False
    goalPoseSet = False
    currentState = 'init'
    nodes = []
    current_Point=[0,0]
    goal_point=[0,0]
    reset()
    while True:
       
        if currentState == 'init':
            print('goal point not yet set')
            fpsClock.tick(10)
        elif currentState == 'goalFound':
            if(count<=epochs):
                currentState = 'buildTree'  #进行下一轮的迭代
                current_Point= initial_Point
                print(step)   #打印每一次迭代移动步数
                step=0        #清零，准备下一次的迭代
                #print("next epochs",current_Point)
                #print(Q_table)
                #while True:
                 #   a=1
                reset()
                pygame.draw.circle(screen, blue,initial_Point, GOAL_RADIUS)
                pygame.draw.circle(screen, green,goal_point, GOAL_RADIUS)
            optimizePhase = True
        elif currentState == 'optimize':
            fpsClock.tick(0.5)
            pass
        elif currentState == 'buildTree':
            count=count+1
            print("epochs   " + str(count))
            while(judge_goalfound(current_Point,goal_point)==False):
                rand=random.random()                                  #产生0-1随机实数
                last_point=current_Point                             #存储上一个点
                #print(last_point)
                if(count>1500):
                    eps=1
                elif(count>1000):
                    eps=0.9
                elif(count>800):
                    eps=0.85
                elif(count>600):
                    eps=0.75
                elif(count>400):
                    eps=0.65
                elif(count>200):
                    eps=0.55
                elif(count>20):
                    eps=0.45
                else:
                    eps=0.4
                if(rand<eps):              
                    current_Point=max_Q_action(current_Point)         #贪婪法则
                    step+=1
                    #print(current_Point)
                else:
                    random_Point=random_action(current_Point)        #随机法则
                    #print(current_Point)
               # while True:
                #        a=1
                    #if(collides(current_Point) == True):
                     #   current_Point=random_Point[1]    #碰到障碍物，返回上一步
                    #else:
                    current_Point=random_Point[1]     #未碰到障碍物，随机走一步
                    step+=1
                   # while collides(current_Point) == True:
                    #   current_Point=random_action(current_Point)    #随机法则
                rew=reward(current_Point,goal_point)                  #回报函数
                if(rew<100):
                  update_Q_table(current_Point,rew)                     #更新Q_table
                else:
                  updata_reward_Q_table(current_Point,rew) 
                t_last_point=tuple(last_point)
                t_current_Point=tuple(current_Point)
                #print(t_last_point)
                #print (rew)
                #print(t_current_Point)
               # while True:
                #       a=1
                pygame.draw.line(screen,white,t_last_point,t_current_Point,2)             # current_Point.point  
                pygame.display.update()                
            currentState = 'goalFound'         
                   
        #handle events
        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                sys.exit("Exiting")
            if e.type == MOUSEBUTTONDOWN:
                print('mouse down')
                if currentState == 'init':
                    if initPoseSet == False:
                       # nodes = []
                        if collides(e.pos) == False:
                            print('initiale pose set: '+str(e.pos))
                            initialPoint = Node(e.pos, None)
                            initial_Point=list(e.pos)
                            #e.pos=(760,200)
                            #current_Point=list(e.pos)     #通过鼠标确定初始点
                            #current_Point=list([760,200])  #设定初始点
                            initial_Point=list(e.pos)
                            current_Point=list(e.pos)
                            print(current_Point)
                            initPoseSet = True
                            pygame.draw.circle(screen, blue, e.pos, GOAL_RADIUS)
                    elif goalPoseSet == False:
                        print('goal pose set: '+str(e.pos))
                        if collides(e.pos) == False:
                            #e.pos=(100,60)  #设定目标点
                            goal_point=list(e.pos)
                            goalPoseSet = True
                            pygame.draw.circle(screen, green,e.pos, GOAL_RADIUS)
                            currentState = 'buildTree'
                            init_Q_table(current_Point,goal_point)  #初始化Q_table
                            #print(Q_table)
                            #while True:
                             #   a=1
                else:
                    currentState = 'init'
                    initPoseSet = False
                    goalPoseSet = False
                    reset()

        pygame.display.update()
        fpsClock.tick(10)


# if python says run, then we should run
if __name__ == '__main__':
    main()
    input("press Enter to quit")
