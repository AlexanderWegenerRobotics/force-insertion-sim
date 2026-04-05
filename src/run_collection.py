import threading
from simcore import RobotSystem, load_yaml, Pose
from utils.sensor_callback import SensorCallback
from task.insertion_task import InsertionTask


def main():
    config = load_yaml("configs/global_config.yaml")
    system = RobotSystem(config)
    
    sensor_cb = SensorCallback(device_name="arm", gravity_compensation=True, ee_mass=0.15, filter_type="butterworth", cutoff_hz=10.0, fs_hz=200.0, butter_order=3)
    #sensor_cb = SensorCallback(device_name="arm", gravity_compensation=True, ee_mass=0.15, filter_type="none")
    system.sim.register_log_callback(sensor_cb)
    system.sensor_cb = sensor_cb

    task_cfg = load_yaml(config.get("task_config"))
    task = InsertionTask(system, task_cfg)
    

    if system.headless:
        system.run()
        try:
            task.run()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            system.stop()
    else:
        task_thread = threading.Thread(target=task.run, daemon=True)
        task_thread.start()

        try:
            system.run()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            system.stop()

if __name__ == "__main__":
    main()