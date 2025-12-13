class BasePlanner:
    def plan(self, start_pose, goal_pose):
        """Plan a trajectory from start_pose to goal_pose."""
        raise NotImplementedError