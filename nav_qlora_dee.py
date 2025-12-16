# Adapt PrismaticEarlyExit for navigation action space (4D: v_x, v_y, yaw_rate, speed)
class PrismaticNavEarlyExit(PrismaticForVisionAndLanguageGeneration):
    def __init__(self, config, exit_layers=[4, 8], exit_threshold=0.95):
        super().__init__(config)
        self.exit_layers = exit_layers
        self.exit_threshold = exit_threshold
        # Navigation action head (4D: x/y velocity, yaw rate, speed)
        self.action_head = nn.Linear(config.hidden_size, 4)  
        # DEE exit branches for navigation
        self.exit_heads = nn.ModuleDict()
        for layer_idx in exit_layers:
            self.exit_heads[str(layer_idx)] = nn.Linear(config.hidden_size, 4)
        
        # Navigation-specific confidence metric (path feasibility + action variance)
        def nav_confidence_fn(action_pred, visual_features):
            # 1. Action variance (lower = more confident)
            action_var = torch.var(action_pred, dim=-1).mean()
            # 2. Object detection confidence (from visual features)
            obj_conf = visual_features[:, :, -1].mean()  # Assume last channel = detection confidence
            # Combine metrics (weighted for navigation)
            return (1 - action_var) * 0.6 + obj_conf * 0.4
        
        self.confidence_fn = nav_confidence_fn