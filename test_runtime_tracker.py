import torch
import torch.nn as nn
import sys
import os

# Add the workspace root to Python path to allow importing models.runtime_tracker
# This assumes the test script is in the root of the MOTIP workspace, 
# or that the models package is otherwise findable.
# Adjust the path if your script is located elsewhere relative to the models directory.
# For example, if test_runtime_tracker.py is in /Users/kerem/MOTIP/tests/
# and runtime_tracker.py is in /Users/kerem/MOTIP/models/
# you might need sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), \'..\')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models.runtime_tracker import RuntimeTracker
from utils.misc import distributed_device # Assuming this utility is available and sets up device
from utils.box_ops import box_cxcywh_to_xywh # Assuming this utility

# Dummy Model
class DummyModel(nn.Module):
    def __init__(self, num_classes=80, num_id_vocabulary=100):
        super().__init__()
        self.num_classes = num_classes
        self.num_id_vocabulary = num_id_vocabulary # Max ID + 1 for newborn
        # Dummy layers to allow attribute access if RuntimeTracker inspects them
        self.trajectory_modeling = nn.Sequential(nn.Linear(10,10)) # Placeholder
        self.trajectory_modeling.motion_extractor = nn.Identity() # Placeholder
        self.trajectory_modeling.kalman_filter = True # Placeholder to signify it exists

    def forward(self, frames=None, seq_info=None, part="detr"):
        device = frames.device if frames is not None else distributed_device()
        if part == "detr":
            # Simulate DETR output
            # N_queries = 10 (number of detections proposed by DETR)
            N_queries = 10 
            # Batch size is 1 for tracker update
            # pred_logits: (batch_size, N_queries, num_classes)
            pred_logits = torch.rand(1, N_queries, self.num_classes, device=device)
            # pred_boxes: (batch_size, N_queries, 4) in cxcywh format, normalized
            pred_boxes = torch.rand(1, N_queries, 4, device=device)
            pred_boxes[:, :, 2:] = torch.sigmoid(pred_boxes[:, :, 2:]) # Ensure w,h are positive-like
            # outputs: (batch_size, N_queries, embed_dim) - e.g., 256
            outputs = torch.rand(1, N_queries, 256, device=device)
            return {
                "pred_logits": pred_logits,
                "pred_boxes": pred_boxes,
                "outputs": outputs
            }
        elif part == "trajectory_modeling":
            # Dummy processing for trajectory modeling part
            # It should return a modified seq_info or some features
            return seq_info # Simplest pass-through
        elif part == "id_decoder":
            # Simulate ID decoder output
            # Based on RuntimeTracker: id_logits = id_logits[0, 0, 0] -> so expect (1,1,1, N_det, N_ids_vocab)
            # N_det comes from unknown_features.shape[3] in the original RuntimeTracker code.
            # unknown_features is (1, 1, T_unknown, N_unknown, C)
            # So, N_det should be seq_info["unknown_features"].shape[3] (N_unknown)
            # N_ids_vocab should be self.num_id_vocabulary
            
            # Based on _get_id_pred_labels:
            # current_features = output_embeds[None, ...] (1, N_det, C)
            # unknown_features shape in seq_info: (1, 1, 1, N_det, C)
            # So, id_logits should be (1, 1, 1, N_det, self.num_id_vocabulary + 1) if it includes newborn class
            # Or (1, 1, 1, N_det, self.num_id_vocabulary) if newborn handled separately
            # The RuntimeTracker code does: `id_logits.softmax(dim=-1)` or `id_logits.sigmoid()`
            # and then `match self.assignment_protocol`
            # The scores are expected to be (N_det, self.num_id_vocabulary + 1) for assignment generally
            # Let\'s assume N_det is the number of current detections
            
            # N_unknown for current detections would be seq_info["unknown_features"].shape[3]
            # Typically, this is called with current detections.
            # In _get_id_pred_labels_with_motion: current_features are (1, N_det, C)
            # seq_info["unknown_features"] = current_features[None, None, ...] giving (1,1,1,N_det,C)
            N_current_detections = seq_info["unknown_features"].shape[3]

            # The id_logits are expected as (batch, num_seq, num_unknown_queries_per_seq, N_det_per_unknown_query, num_id_classes)
            # In RuntimeTracker, it\'s indexed [0,0,0] which implies batch, seq, unknown_quries_per_seq are all 1.
            # So, (1, 1, 1, N_current_detections, self.num_id_vocabulary + 1) where +1 is for newborn class.
            # (The original code sometimes uses num_id_vocabulary and sometimes implies num_id_vocabulary+1 in assignment scores)
            # For safety, let\'s make it num_id_vocabulary + 1, where the last class is "newborn"
            id_logits = torch.rand(1, 1, 1, N_current_detections, self.num_id_vocabulary + 1, device=device)
            
            # The model might also return other things, but RuntimeTracker only uses id_logits here.
            return id_logits, None, None 
        return {}

def main():
    print(f"Torch version: {torch.__version__}")
    device = distributed_device() # Uses CUDA if available, else CPU
    print(f"Using device: {device}")

    # Tracker Parameters
    sequence_hw = (720, 1280) # Example H, W
    miss_tolerance = 10 # Shorter for quick testing
    det_thresh = 0.3    # Lower detection threshold for testing
    newborn_thresh = 0.4 # Threshold for a detection to be considered newborn if not matched
    id_thresh = 0.2     # Threshold for ID assignment confidence
    num_ids_vocabulary = 20 # Smaller ID vocabulary for testing

    # Initialize Dummy Model
    dummy_model = DummyModel(num_id_vocabulary=num_ids_vocabulary).to(device)
    dummy_model.eval()

    # Initialize RuntimeTracker
    print("Initializing RuntimeTracker...")
    tracker = RuntimeTracker(
        model=dummy_model,
        sequence_hw=sequence_hw,
        use_sigmoid=False, # Or True, depending on model output style for IDs
        assignment_protocol="hungarian", # or "object-max", "id-max"
        miss_tolerance=miss_tolerance,
        det_thresh=det_thresh,
        newborn_thresh=newborn_thresh,
        id_thresh=id_thresh,
        area_thresh=0,
        only_detr=False,
        dtype=torch.float32,
        use_motion=True,       # Enable motion
        motion_weight=0.3      # Example motion weight
    )
    print("RuntimeTracker initialized.")

    # Simulate a few frames
    num_frames_to_test = 5
    print(f"Simulating {num_frames_to_test} frames...")
    for frame_idx in range(num_frames_to_test):
        print(f"--- Frame {frame_idx + 1} ---")
        # Dummy image tensor (Batch, Channels, Height, Width)
        # The tracker expects model(frames=image), so image should be what model expects.
        # Typically B,C,H,W
        dummy_image_input = torch.rand(1, 3, sequence_hw[0], sequence_hw[1], device=device)
        
        try:
            tracker.update(image=dummy_image_input)
            current_results = tracker.get_track_results()
            
            print(f" Detections found: {len(current_results.get('id', []))}")
            if len(current_results.get('id', [])) > 0:
                print(f"  Scores: {current_results['score']}")
                print(f"  Categories: {current_results['category']}")
                print(f"  BBoxes (unnormalized xywh): {current_results['bbox']}")
                print(f"  IDs: {current_results['id']}")
            
            # Optional: Inspect some internal trajectory states if needed for debugging
            # print(f"  Trajectory boxes shape: {tracker.trajectory_boxes.shape}")
            # if tracker.use_motion and hasattr(tracker, 'trajectory_velocities'):
            #     print(f"  Trajectory velocities shape: {tracker.trajectory_velocities.shape}")

        except Exception as e:
            print(f"ERROR during tracker.update() on frame {frame_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            break 
        
        if frame_idx < num_frames_to_test - 1:
            # Simulate some time passing, not strictly necessary for this test
            pass

    print("\nTest simulation finished.")
    print("Check for errors above. If no errors, basic execution with motion modeling is likely working.")

if __name__ == "__main__":
    # A simple way to ensure utils and models can be found if test script is in root
    # and utils/models are subdirectories.
    # If your project structure is different, you might need to adjust sys.path more.
    # Example: if this script is in /Users/kerem/MOTIP and your modules are in /Users/kerem/MOTIP/models etc.
    # then the initial sys.path.append(os.path.abspath(os.path.dirname(__file__))) should make `from models...` work.
    
    # For utils.misc and utils.box_ops, we need to ensure they exist or provide stubs.
    # Let\'s create dummy utils if they are not found, for the sake of this test script.
    
    try:
        from utils.misc import distributed_device
    except ImportError:
        print("Warning: utils.misc.distributed_device not found. Using a dummy version.")
        def distributed_device():
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Make it available in the global scope for the test
        globals()["distributed_device"] = distributed_device

    try:
        from utils.box_ops import box_cxcywh_to_xywh
    except ImportError:
        print("Warning: utils.box_ops.box_cxcywh_to_xywh not found. Using a dummy version.")
        def box_cxcywh_to_xywh(x):
            # x: (N, 4) tensor with (cx, cy, w, h)
            # returns (N, 4) tensor with (x1, y1, w, h) where x1,y1 is top-left
            if not isinstance(x, torch.Tensor) or x.dim() != 2 or x.shape[-1] != 4:
                 # Simplified handling for the test case where it might receive BxNx4 from model
                if x.dim() == 3 and x.shape[0] == 1 and x.shape[-1] == 4: # B=1
                    x_ = x.squeeze(0)
                else:
                    raise ValueError("Input to dummy box_cxcywh_to_xywh expects (N,4) or (1,N,4)")
            else:
                x_ = x
            x_c, y_c, w, h = x_[..., 0], x_[..., 1], x_[..., 2], x_[..., 3]
            x1 = x_c - w / 2
            y1 = y_c - h / 2
            return torch.stack((x1, y1, w, h), dim=-1)
        globals()["box_cxcywh_to_xywh"] = box_cxcywh_to_xywh
        
    # Ensure models.misc.get_model is available or provide a dummy
    try:
        from models.misc import get_model
    except ImportError:
        print("Warning: models.misc.get_model not found. Using a dummy version.")
        class DummyModelForGetModel:
            def __init__(self, num_id_vocabulary):
                self.num_id_vocabulary = num_id_vocabulary
        
        def get_model(model_instance):
            if hasattr(model_instance, 'num_id_vocabulary'):
                return model_instance 
            else: 
                return DummyModelForGetModel(num_id_vocabulary=20) 

        if 'models' not in sys.modules:
            models_module = type(sys)('models')
            sys.modules['models'] = models_module
            models_misc_module = type(sys)('models.misc')
            sys.modules['models.misc'] = models_misc_module
            setattr(models_misc_module, 'get_model', get_model)
        elif 'models.misc' not in sys.modules:
            models_misc_module = type(sys)('models.misc')
            sys.modules['models.misc'] = models_misc_module
            setattr(models_misc_module, 'get_model', get_model)
        else:
            setattr(sys.modules['models.misc'], 'get_model', get_model)

    main() 