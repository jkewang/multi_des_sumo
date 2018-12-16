import os,sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable SUMO_HOME")

config_path = "/home/jkwang/learn_sumo/quickstart/quickstart.sumo.cfg"
sumoBinary = "/usr/bin/sumo"
sumoguiBinary = "/usr/bin/sumo-gui"
sumoCmd = [sumoguiBinary,"-c",config_path,"--collision.action","remove","--start","--no-step-log","--no-warnings","--no-duration-log"]

import traci
import traci.constants as tc
import math
import numpy as np
import random

class TrafficEnv(object):

    def __init__(self):
        traci.start(sumoCmd)
        self.cross_mapping={
            "-gneE0": "cross_0",
            "gneE0": "cross_3",
            "-gneE1": "cross_3",
            "gneE1": "cross_6",
            "-gneE2": "cross_6",
            "gneE2": "cross_7",
            "-gneE3": "cross_7",
            "gneE3": "cross_8",
            "-gneE4": "cross_8",
            "gneE4": "cross_5",
            "-gneE5": "cross_5",
            "gneE5": "cross_2",
            "-gneE6": "cross_2",
            "gneE6": "cross_1",
            "-gneE7": "cross_1",
            "gneE7": "cross_0",
            "-gneE8": "cross_3",
            "gneE8": "cross_4",
            "-gneE9": "cross_4",
            "gneE9": "cross_7",
            "-gneE10": "cross_4",
            "gneE10": "cross_5",
            "-gneE11": "cross_4",
            "gneE11": "cross_1"
        }

        self.trafficPos_mapping={
            "cross_0": [-1000, 1000],
            "cross_1": [0, 1000],
            "cross_2": [1000, 1000],
            "cross_3": [-1000,0],
            "cross_4": [0,0],
            "cross_5": [1000,0],
            "cross_6": [-1000,-1000],
            "cross_7": [0,-1000],
            "cross_8": [1000,-1000]
        }

        #Env --lanechange.duration
        self.step_num = 0
        self.AgentId = "agent"
        self.VehicleIds = []
        self.TotalReward = 0
        self.StartTime = 0
        self.end = 0
        #traci.vehicle.add("agent", "agent_route")
        traci.vehicle.setColor("agent", (255 , 0, 0, 255))
        traci.vehicle.setSpeed("agent",10)
        traci.gui.trackVehicle('View #0', "agent")

        #States
        self.Route = traci.vehicle.getRoute(self.AgentId)
        self.OccMapState = np.zeros((20, 7))
        self.VehicleState = [0,0,0]
        self.RoadState = [0 for i in range(9)]
        self.state = None
        self.lastTlsTd = "cross_6"

        #property to simulate
        self.end_x = 0
        self.end_y = 1000
        self.AgentX = 0
        self.AgentY = 0
        self.AgentSpeed = 10
        self.AgentAccRate = 2.0
        self.AgentDecRate = 1.0
        self.minLaneNumber = 0
        self.maxLaneNumber = 1
        self.oldDistance = 0
        self.nowDistance = 0

    def reset(self):
        self.end = 0
        self.TotalReward = 0
        self.oldDistance = 0
        self.nowDistance = 0
        self.lastTlsTd = "cross_6"
        self.lastdistance = 0.99
        self.x_v = 0
        self.y_v = 0

        traci.load(["-c",config_path,"--collision.action","remove","--no-step-log","--no-warnings","--no-duration-log"])
        print("Resetting...")
        #traci.vehicle.add("agent", "agent_route")

        random_direct = random.randint(0,1)
        random_edge = random.randint(3,12)
        if random_direct == 0:
            direct = '-'
        else:
            direct = ''
        if random_edge != 12:
            edge = str(random_edge)
        else:
            edge = '0'
        dest_name = direct + "gneE" + edge
        print(dest_name)

        pos = self.trafficPos_mapping[self.cross_mapping[dest_name]]
        self.end_x,self.end_y = pos[0],pos[1]

        traci.vehicle.changeTarget("agent",dest_name)
        self.Route = traci.vehicle.getRoute(self.AgentId)
        print(self.Route)

        traci.vehicle.setColor("agent", (255, 0, 0, 255))
        traci.vehicle.setSpeed("agent", 10)
        traci.gui.trackVehicle('View #0', "agent")

        traci.simulationStep()
        AgentAvailable = False
        while AgentAvailable == False:
            traci.simulationStep()
            self.VehicleIds = traci.vehicle.getIDList()
            if self.AgentId in self.VehicleIds:
                AgentAvailable = True
                self.StartTime = traci.simulation.getCurrentTime()
        for vehId in self.VehicleIds:
            traci.vehicle.subscribe(vehId,(tc.VAR_SPEED,tc.VAR_POSITION,tc.VAR_LANE_INDEX,tc.VAR_DISTANCE))
            traci.vehicle.subscribeLeader(self.AgentId,50)
            if vehId == self.AgentId:
                print("remeber")
                #traci.vehicle.setSpeedMode(self.AgentId,0)
                #traci.vehicle.setLaneChangeMode(self.AgentId,0)
        self.state,breaklight,breakstop,wronglane = self.perception()

        return self.state

    def step(self,action):
        # define action:
        # action  |     meaning
        #    0    |    go straight
        #    1    |    break down
        #    2    |    change left
        #    3    |    change right
        #    4    |    do nothing

        position = traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_POSITION]
        if (abs(position[0])+abs(position[1]))<999:
            self.maxLaneNumber = 2
        else:
            self.maxLaneNumber = 1

        self.end = 0
        reward = 0
        DistanceTravelled = 0
        if action == 0:
            maxSpeed = 16
            time = (maxSpeed - (traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_SPEED])) / self.AgentAccRate
            traci.vehicle.slowDown(self.AgentId, maxSpeed, time)
        elif action == 1:
            time = ((traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_SPEED]) - 0)/self.AgentDecRate
            traci.vehicle.slowDown(self.AgentId, 0, time)
        elif action == 2:
            laneindex = traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_LANE_INDEX]
            if laneindex < self.maxLaneNumber:
                traci.vehicle.changeLane(self.AgentId,laneindex+1,100)
        elif action == 3:
            laneindex = traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_LANE_INDEX]
            if laneindex > self.minLaneNumber:
                traci.vehicle.changeLane(self.AgentId,laneindex-1,100)
        elif action == 4:
            traci.vehicle.setSpeed(self.AgentId,traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_SPEED])
        traci.simulationStep()
        self.VehicleIds = traci.vehicle.getIDList()

        if self.AgentId in self.VehicleIds:
            for vehId in self.VehicleIds:
                traci.vehicle.subscribe(vehId,(tc.VAR_SPEED,tc.VAR_POSITION,tc.VAR_LANE_INDEX,tc.VAR_DISTANCE))
                traci.vehicle.subscribeLeader(self.AgentId,50)

            Vehicle_Params = traci.vehicle.getSubscriptionResults(self.AgentId)
            self.AutocarSpeed = Vehicle_Params[tc.VAR_SPEED]
            posAutox = Vehicle_Params[tc.VAR_POSITION]
            if math.sqrt((self.end_x-posAutox[0])**2+(self.end_y-posAutox[1])**2)<30:
                self.end = 100

            self.state,breaklight,breakstop,wronglane = self.perception()
            reward = self.cal_reward(self.end,breaklight,breakstop,wronglane)
        else:
            #self.state = self.perception()
            self.end = 1
            reward = self.cal_reward(is_collision=self.end,breaklight=0,breakstop=0,wronglane=0)
            DistanceTravelled = 0


        return self.state, reward, self.end, DistanceTravelled

    def cal_reward(self,is_collision,breaklight,breakstop,wronglane):
        if is_collision == 1:
            print("collision!")
            return -30
        elif is_collision == 100:
            print("arrive!")
            return 100
        else:
            self.nowDistance = traci.vehicle.getDistance(self.AgentId)
            del_distance = self.nowDistance - self.oldDistance
            reward = float(del_distance-8)/500.0
            self.oldDistance = self.nowDistance
            if breaklight == 1:
                reward -= 4
            if breakstop == 1:
                reward -= 1
            if wronglane == 1:
                print("wronglane!")
                reward -= 2
            return reward

    def perception(self):

        #the state is defined as:
        # 0   | 1    | 2    | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
        #speed|cos(a)|sin(a)|l? |r? |dis| r | y | g |l? | c? | r? |
        #

        self.VehicleIds = traci.vehicle.getIDList()

        AllVehicleParams = []

        #----------------------------to get the vehicle state------------------------
        for vehId in self.VehicleIds:
            traci.vehicle.subscribe(vehId, (tc.VAR_SPEED, tc.VAR_POSITION, tc.VAR_ANGLE, tc.VAR_LANE_INDEX, tc.VAR_DISTANCE, tc.VAR_LANE_ID))
            VehicleParam = traci.vehicle.getSubscriptionResults(vehId)
            #AllVehicleParams.append(vehId)
            if vehId != self.AgentId:
                AllVehicleParams.append(VehicleParam)
            else:
                self.AgentSpeed = VehicleParam[tc.VAR_SPEED]
                self.AgentAngle = (VehicleParam[tc.VAR_ANGLE]/180)*math.pi
                self.AgentX = VehicleParam[tc.VAR_POSITION][0]
                self.AgentY = VehicleParam[tc.VAR_POSITION][1]
        self.VehicleState = [self.AgentSpeed,math.cos(self.AgentAngle),math.sin(self.AgentAngle)]

        #---------------------to calculate the occupanied state-----------------------
        LOW_X_BOUND = -6
        HIGH_X_BOUND = 6
        LOW_Y_BOUND = -10
        HIGH_Y_BOUND = 30
        self.OccMapState = np.zeros((20, 7))
        for VehicleParam in AllVehicleParams:
            VehiclePos = VehicleParam[tc.VAR_POSITION]
            rol = math.sqrt((VehiclePos[0]-self.AgentX)**2+(VehiclePos[1]-self.AgentY)**2)
            theta = math.atan2(VehiclePos[1]-self.AgentY,VehiclePos[0]-self.AgentX)
            reltheta = theta + self.AgentAngle
            relX = rol*math.cos(reltheta)
            relY = rol*math.sin(reltheta)
            if (relX>LOW_X_BOUND and relX<HIGH_X_BOUND) and (relY>LOW_Y_BOUND and relY<HIGH_Y_BOUND):
                indexX = int((6 + relX)/2 - 0.5)
                indexY = int((30 - relY)/2 - 0.5)
                self.OccMapState[indexY,indexX] = 1.0

            #add for fc dqn
        self.OccMapState = self.OccMapState.reshape(-1)

        #-------------------------------to get the RoadState----------------------------
        #RoadState: [leftcan rightcan distance r y g leftava centerava rightava]
        self.RoadState = [1.0 for i in range(9)]
        now_laneindex = 0
        for vehId in self.VehicleIds:
            if vehId == self.AgentId:
                now_laneindex = traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_LANE_INDEX]
                now_roadid = traci.vehicle.getRoadID(self.AgentId)
        if now_laneindex + 1 > self.maxLaneNumber:
            self.RoadState[0] = 0
        elif now_laneindex - 1 < self.minLaneNumber:
            self.RoadState[1] = 0

        breaklight = 0
        breakstop = 0

        next_roadid = ""
        for i in range(len(self.Route)):
            if now_roadid == self.Route[i] and (i != len(self.Route)-1):
                next_roadid = self.Route[i+1]

        print("now_road:",now_roadid," next_road:",next_roadid)
        try:
            nextTlsId = self.cross_mapping[now_roadid]
            rygState = traci.trafficlight.getRedYellowGreenState(nextTlsId)
            links = traci.trafficlight.getControlledLinks(nextTlsId)
            index = 0
            print(links)
            for i in range(len(links)):
                str_1 = links[i][0][0]
                edge_1 = str_1[0:str_1.rfind("_")]
                str_2 = links[i][0][1]
                edge_2 = str_2[0:str_2.rfind("_")]
                if edge_1 == now_roadid and edge_2 == next_roadid:
                    index = i
                    break
            nextLight = rygState[index]
            x,y = self.trafficPos_mapping[nextTlsId][0],self.trafficPos_mapping[nextTlsId][1]
            x_v,y_v = traci.vehicle.getPosition(self.AgentId)
            #print("trying")
            distance = math.sqrt((x_v-x)**2+(y_v-y)**2)/1000
            #print("distance=",distance)
            if distance > 0.1 and distance==self.lastdistance:
                #print("breakstop")
                breakstop = 1
            self.lastTlsTd = nextTlsId
            self.lastdistance = distance
            #print("last_distance",self.lastdistance)
            #print(nextTlsId)
            #print(distance)
        except:
            x_v, y_v = traci.vehicle.getPosition(self.AgentId)
            if ((self.x_v-x_v)==0 and (self.y_v-y_v)==0):
                breakstop = 1
            else:
                breakstop = 0
            rygState = traci.trafficlight.getRedYellowGreenState(self.lastTlsTd)
            links = traci.trafficlight.getControlledLinks(self.lastTlsTd)
            index = 0
            for i in range(len(links)):
                str_1 = links[i][0][0]
                edge_1 = str_1[0:str_1.rfind("_")]
                str_2 = links[i][0][1]
                edge_2 = str_2[0:str_2.rfind("_")]
                if edge_1 == now_roadid and edge_2 == next_roadid:
                    index = i
                    break
            nextLight = rygState[index]

            #nextLight='g'
            distance = 100
            if nextLight == ('r' or 'R'):
                breaklight = 1
            self.x_v,self.y_v = x_v,y_v

            #print("except")

        self.RoadState[2] = distance
        if nextLight == 'g' or nextLight == 'G':
            self.RoadState[3] = 0
            self.RoadState[4] = 0
            self.RoadState[5] = 1
        elif nextLight == 'y' or nextLight == 'Y':
            self.RoadState[3] = 0
            self.RoadState[4] = 1
            self.RoadState[5] = 0
        else:
            self.RoadState[3] = 1
            self.RoadState[4] = 0
            self.RoadState[5] = 0

        for vehId in self.VehicleIds:
            if vehId == self.AgentId:
                nowLaneId = traci.vehicle.getSubscriptionResults(self.AgentId)[tc.VAR_LANE_ID]
                links = traci.lane.getLinks(nowLaneId)
                self.RoadState[6] = 0
                self.RoadState[7] = 0
                self.RoadState[8] = 0
                for link in links:
                    okRoad = link[0][0:link[0].rfind("_")]
                    if okRoad in self.Route:
                        self.RoadState[7] = 1
                try:
                    leftLaneId = nowLaneId[0:-1] + str(int(nowLaneId[-1])+1)
                    links = traci.lane.getLinks(leftLaneId)
                    for link in links:
                        okRoad = link[0][0:link[0].rfind("_")]
                        if okRoad in self.Route:
                            self.RoadState[6] = 1
                except:
                    self.RoadState[6] = 0
                try:
                    rightLaneId = nowLaneId[0:-1] + str(int(nowLaneId[-1])-1)
                    links = traci.lane.getLinks(rightLaneId)
                    for link in links:
                        okRoad = link[0][0:link[0].rfind("_")]
                        if okRoad in self.Route:
                            self.RoadState[8] = 1
                except:
                    self.RoadState[8] = 0
        wronglane = 0
        if distance <0.1 and self.RoadState[7]==0:
            wronglane = 1

        return [self.OccMapState,self.VehicleState,self.RoadState],breaklight,breakstop,wronglane

