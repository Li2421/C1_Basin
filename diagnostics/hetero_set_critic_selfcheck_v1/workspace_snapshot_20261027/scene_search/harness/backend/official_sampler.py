import numpy as np

class OfficialSampler:

    def set_random_ini(self):
            Num = self.Num.value()
            r_min = self.r_min.value()
            
            if Num:
                for times in range(100):
    
                    ini_x=[]
    
                    for i in range(Num):
    
                        for j in range(1000): 
                        
                            ini=np.random.rand(2)*np.array([2,2])
                            RIGHT=True
    
                            for k in range(len(ini_x)):
                                if(np.linalg.norm(ini-ini_x[k])<r_min+0.05):
                                    RIGHT=False 
                            
                            if RIGHT:
                                ini_x+=[ini]
                                break 
    
                    if len(ini_x)==Num:
                        print("get intial positions")
                        break 
    
                    if (times+1)%10==0:
                        print('Try %s times, cannot find. Keep trying.'%(times+1))
    
                if times == 99:
                    print('cannot find initial positions, please decrease the number of robots')
            else:
    
                print("Please set the number of robots")
    
            self.ini_x=ini_x 
    
            self.ini_v=[]
            for i in range(Num):
                self.ini_v+=[np.zeros(2)]

    def set_random_target(self):
            r_min=self.r_min.value()
            Num = self.Num.value()
            epsilon = self.epsilon.value()
            if Num:
                for times in range(100):
    
                    target=[]
    
                    for i in range(Num):
    
                        for j in range(1000): 
                        
                            tar=np.random.rand(2)*np.array([2,2])
                            RIGHT=True
    
                            for k in range(len(target)):
                                if(np.linalg.norm(tar-target[k])<r_min+2*epsilon):
                                    RIGHT=False 
                            
                            if RIGHT:
                                target+=[tar]
                                break
    
                    if len(target)==Num:
                        print("get target positions")
                        break
                    if (times+1)%10==0:
                        print('Try %s times, cannot find. Keep trying.'%(times+1))
                if times == 99:
                    print('cannot find target positions, please decrease the number of robots')
            else:
                
                print("Please set the number of robots")
    
            self.goal=target
