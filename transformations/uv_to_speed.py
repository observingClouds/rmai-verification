import numpy as np
class UVToSpeed():
    def __init__(self,trans_dict):
        self.trans_dict = trans_dict    

    def execute(self,input_ds):
        u_wind = self.trans_dict.get("u","10u")
        v_wind = self.trans_dict.get("v","10v")
        wind_speed = self.trans_dict.get("speed","10s")
        speed = np.sqrt(input_ds[u_wind]**2 + input_ds[v_wind]**2)
        input_ds[wind_speed] = speed
        return input_ds
