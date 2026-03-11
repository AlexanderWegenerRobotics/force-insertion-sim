import threading
from simcore import RobotSystem, load_yaml
from utils.sensor_callback import sensor_log_callback
from task.insertion_task import InsertionTask
from functools import partial

def main():
    config = load_yaml("configs/global_config.yaml")
    system = RobotSystem(config)
    system.sim.register_log_callback(partial(sensor_log_callback, device_name="arm"))

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