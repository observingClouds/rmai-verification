import numpy as np
class UVToSpeed():
    def __init__(self, u : str = "10u", v : str = "10v", speed : str = "10s"):
        self.u_wind = u
        self.v_wind = v
        self.wind_speed = speed

    def execute(self,input_ds):
        speed = np.sqrt(input_ds[self.u_wind]**2 + input_ds[self.v_wind]**2)
        input_ds[self.wind_speed] = speed
        return input_ds
