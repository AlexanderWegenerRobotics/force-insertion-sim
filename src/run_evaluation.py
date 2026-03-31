import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import threading
from simcore import RobotSystem, load_yaml
from utils.sensor_callback import SensorCallback
from policy.learned_task import LearnedTask


def main():
    config = load_yaml("configs/global_config.yaml")
    system = RobotSystem(config)

    sensor_cb = SensorCallback(device_name="arm", gravity_compensation=True, ee_mass=0.15)
    system.sim.register_log_callback(sensor_cb)

    task_cfg = load_yaml(config.get("task_config"))
    policy_cfg = load_yaml(config.get("policy_config"))
    task = LearnedTask(system=system, task_cfg=task_cfg, policy_cfg=policy_cfg)

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