from simcore import RobotSystem, load_yaml, Pose
from utils.sensor_callback import sensor_log_callback
from functools import partial
import threading

def main():
    config = load_yaml("configs/global_config.yaml")

    system = RobotSystem(config)
    system.sim.register_log_callback(partial(sensor_log_callback, device_name="arm"))
    
    task_thread = threading.Thread(
        target=run_task,
        args=(system,),
        daemon=True
    )
    task_thread.start()
    
    system.set_controller_mode("arm", "impedance")

    target = Pose(position=[0.5, 0.0, 0.8], quaternion=[0, 1, 0, 0])
    system.set_target("arm", {"x": target})

    try:
        system.run()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        system.stop()


if __name__ == "__main__":
    main()
