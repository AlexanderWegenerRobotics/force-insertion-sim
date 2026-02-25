def sensor_log_callback(mj_model, mj_data, device_name="arm"):
    p = f"{device_name}/"
    return {
        'sensors': {
            'ft_force':    mj_data.sensor(f'{p}ft_force').data.copy(),
            'ft_torque':   mj_data.sensor(f'{p}ft_torque').data.copy(),
            'ee_linvel':   mj_data.sensor(f'{p}ee_linvel').data.copy(),
            'ee_angvel':   mj_data.sensor(f'{p}ee_angvel').data.copy(),
            'peg_tip_pos': mj_data.sensor(f'{p}peg_tip_pos').data.copy(),
        }
    }