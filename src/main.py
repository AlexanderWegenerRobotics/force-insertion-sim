import threading
from simcore import RobotSystem, load_yaml
from utils.sensor_callback import SensorCallback
from task.insertion_task import InsertionTask

def main():
    config = load_yaml("configs/global_config.yaml")
    system = RobotSystem(config)
    
    sensor_cb = SensorCallback(device_name="arm", gravity_compensation=True, ee_mass=0.15)
    system.sim.register_log_callback(sensor_cb)

    task_cfg = load_yaml(config.get("task_config"))
    task = InsertionTask(system, task_cfg)

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