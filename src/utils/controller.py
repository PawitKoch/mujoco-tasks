class BaseController:
    def execute_trajectory(self, trajectory):
        """Execute the given trajectory."""
        raise NotImplementedError
    
    def control(self, current_pose, target_pose):
        """Compute control action to move from current_pose to target_pose."""
        raise NotImplementedError
