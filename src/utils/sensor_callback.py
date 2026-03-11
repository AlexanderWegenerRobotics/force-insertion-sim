import numpy as np

class SensorCallback:
    def __init__(self, device_name: str, gravity_compensation: bool = False, 
                 ee_mass: float = 0.0):
        self.device_name = device_name
        self.p = f"{device_name}/"
        self.gravity_compensation = gravity_compensation
        self.ee_mass = ee_mass  # total mass of EE + peg in kg

    def __call__(self, mj_model, mj_data):
        p = self.p
        ft_force = mj_data.sensor(f'{p}ft_force').data.copy()
        ft_torque = mj_data.sensor(f'{p}ft_torque').data.copy()

        if self.gravity_compensation:
            # Get EE orientation from MuJoCo to rotate gravity into sensor frame
            site_id = mj_model.site(f'{p}ft_sensor').id
            R_world_sensor = mj_data.site_xmat[site_id].reshape(3, 3)
            g_world = np.array([0, 0, self.ee_mass * 9.81])
            g_sensor = R_world_sensor.T @ g_world
            ft_force -= g_sensor

        return {
            'sensors': {
                'ft_force':    ft_force,
                'ft_torque':   ft_torque,
                'ee_linvel':   mj_data.sensor(f'{p}ee_linvel').data.copy(),
                'ee_angvel':   mj_data.sensor(f'{p}ee_angvel').data.copy(),
                'peg_tip_pos': mj_data.sensor(f'{p}peg_tip_pos').data.copy(),
            }
        }