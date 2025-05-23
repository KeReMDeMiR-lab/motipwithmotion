# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import einops
from scipy.optimize import linear_sum_assignment

from structures.instances import Instances
from structures.ordered_set import OrderedSet
from utils.misc import distributed_device
from utils.box_ops import box_cxcywh_to_xywh
from models.misc import get_model


class RuntimeTracker:
    def __init__(
            self,
            model,
            # Sequence infos:
            sequence_hw: tuple,
            # Inference settings:
            use_sigmoid: bool = False,
            assignment_protocol: str = "hungarian",
            miss_tolerance: int = 30,
            det_thresh: float = 0.5,
            newborn_thresh: float = 0.5,
            id_thresh: float = 0.1,
            area_thresh: int = 0,
            only_detr: bool = False,
            dtype: torch.dtype = torch.float32,
            # Motion modeling parameters
            use_motion: bool = False,
            motion_weight: float = 0.5,
    ):
        self.model = model
        self.model.eval()

        self.dtype = dtype

        # For FP16:
        if self.dtype != torch.float32:
            if self.dtype == torch.float16:
                self.model.half()
            else:
                raise NotImplementedError(f"Unsupported dtype {self.dtype}.")

        self.use_sigmoid = use_sigmoid
        self.assignment_protocol = assignment_protocol.lower()
        self.miss_tolerance = miss_tolerance
        self.det_thresh = det_thresh
        self.newborn_thresh = newborn_thresh
        self.id_thresh = id_thresh
        self.area_thresh = area_thresh
        self.only_detr = only_detr
        self.num_id_vocabulary = get_model(model).num_id_vocabulary
        
        # Motion modeling parameters
        self.use_motion = use_motion
        self.motion_weight = motion_weight

        # Check for the legality of settings:
        assert self.assignment_protocol in ["hungarian", "id-max", "object-max", "object-priority", "id-priority"], \
            f"Assignment protocol {self.assignment_protocol} is not supported."

        self.bbox_unnorm = torch.tensor(
            [sequence_hw[1], sequence_hw[0], sequence_hw[1], sequence_hw[0]],
            dtype=dtype,
            device=distributed_device(),
        )

        # Trajectory fields:
        self.next_id = 0
        self.id_label_to_id = {}
        self.id_queue = OrderedSet()
        # Init id_queue:
        for i in range(self.num_id_vocabulary):
            self.id_queue.add(i)
        # All fields are in shape (T, N, ...)
        self.trajectory_features = torch.zeros(
            (0, 0, 256), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_boxes = torch.zeros(
            (0, 0, 4), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_id_labels = torch.zeros(
            (0, 0), dtype=torch.int64, device=distributed_device(),
        )
        self.trajectory_times = torch.zeros(
            (0, 0), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_masks = torch.zeros(
            (0, 0), dtype=torch.bool, device=distributed_device(),
        )
        
        # Motion-specific fields
        if self.use_motion:
            self.trajectory_velocities = torch.zeros(
                (0, 0, 4), dtype=dtype, device=distributed_device(),
            )
            self.trajectory_accelerations = torch.zeros(
                (0, 0, 4), dtype=dtype, device=distributed_device(),
            )
            self.kalman_states = {}  # Store Kalman filter states
            self.kalman_covariances = {}  # Store Kalman filter covariances

        self.current_track_results = {}
        return

    @torch.no_grad()
    def update(self, image):
        detr_out = self.model(frames=image, part="detr")
        scores, categories, boxes, output_embeds = self._get_activate_detections(detr_out=detr_out)
        if self.only_detr:
            id_pred_labels = self.num_id_vocabulary * torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device)
        else:
            # Use motion-aware ID prediction if enabled
            if self.use_motion and hasattr(self.model, 'trajectory_modeling') and hasattr(self.model.trajectory_modeling, 'motion_extractor'):
                id_pred_labels = self._get_id_pred_labels_with_motion(boxes=boxes, output_embeds=output_embeds)
            else:
                id_pred_labels = self._get_id_pred_labels(boxes=boxes, output_embeds=output_embeds)
                
        # Filter out illegal newborn detections:
        keep_idxs = (id_pred_labels != self.num_id_vocabulary) | (scores > self.newborn_thresh)
        scores = scores[keep_idxs]
        categories = categories[keep_idxs]
        boxes = boxes[keep_idxs]
        output_embeds = output_embeds[keep_idxs]
        id_pred_labels = id_pred_labels[keep_idxs]

        # A hack implementation, before assign new id labels, update the id_queue to ensure the uniqueness of id labels:
        n_activate_id_labels = 0
        n_newborn_targets = 0
        for _ in range(len(id_pred_labels)):
            if id_pred_labels[_].item() != self.num_id_vocabulary:
                n_activate_id_labels += 1
                self.id_queue.add(id_pred_labels[_].item())
            else:
                n_newborn_targets += 1

        # Make sure the length of newborn instances is less than the length of remaining IDs:
        n_remaining_ids = len(self.id_queue) - n_activate_id_labels
        if n_newborn_targets > n_remaining_ids:
            keep_idxs = torch.ones(len(id_pred_labels), dtype=torch.bool, device=id_pred_labels.device)
            newborn_idxs = (id_pred_labels == self.num_id_vocabulary)
            newborn_keep_idxs = torch.ones(len(newborn_idxs), dtype=torch.bool, device=newborn_idxs.device)
            newborn_keep_idxs[n_remaining_ids:] = False
            keep_idxs[newborn_idxs] = newborn_keep_idxs
            scores = scores[keep_idxs]
            categories = categories[keep_idxs]
            boxes = boxes[keep_idxs]
            output_embeds = output_embeds[keep_idxs]
            id_pred_labels = id_pred_labels[keep_idxs]
        pass

        # Assign new id labels:
        id_labels = self._assign_newborn_id_labels(pred_id_labels=id_pred_labels)

        if len(torch.unique(id_labels)) != len(id_labels):
            print(id_labels, id_labels.shape)
            exit(-1)

        # Update the results:
        self.current_track_results = {
            "score": scores,
            "category": categories,
            "bbox": box_cxcywh_to_xywh(boxes) * self.bbox_unnorm,
            "id": torch.tensor(
                [self.id_label_to_id[_] for _ in id_labels.tolist()], dtype=torch.int64,
            ),
        }

        # Update id_queue:
        for _ in range(len(id_labels)):
            self.id_queue.add(id_labels[_].item())

        # Update trajectory infos:
        self._update_trajectory_infos(boxes=boxes, output_embeds=output_embeds, id_labels=id_labels)

        # Filter out inactive tracks:
        self._filter_out_inactive_tracks()
        pass
        return

    def _get_id_pred_labels_with_motion(self, boxes: torch.Tensor, output_embeds: torch.Tensor):
        """
        Enhanced ID prediction that incorporates motion information
        """
        if self.trajectory_features.shape[0] == 0:
            return self.num_id_vocabulary * torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device)
        else:
            # 1. prepare current infos:
            current_features = output_embeds[None, ...]     # (T, N, ...)
            current_boxes = boxes[None, ...]                # (T, N, 4)
            current_masks = torch.zeros((1, output_embeds.shape[0]), dtype=torch.bool, device=distributed_device())
            current_times = self.trajectory_times.shape[0] * torch.ones(
                (1, output_embeds.shape[0]), dtype=torch.int64, device=distributed_device(),
            )
            
            # 2. prepare seq_info with motion features:
            seq_info = {
                "trajectory_features": self.trajectory_features[None, None, ...],
                "trajectory_boxes": self.trajectory_boxes[None, None, ...],
                "trajectory_id_labels": self.trajectory_id_labels[None, None, ...],
                "trajectory_times": self.trajectory_times[None, None, ...],
                "trajectory_masks": self.trajectory_masks[None, None, ...],
                "unknown_features": current_features[None, None, ...],
                "unknown_boxes": current_boxes[None, None, ...],
                "unknown_masks": current_masks[None, None, ...],
                "unknown_times": current_times[None, None, ...],
            }
            
            # Add motion predictions using Kalman filter
            if hasattr(self, 'kalman_states'):
                motion_predictions = self._get_motion_predictions(boxes)
                if motion_predictions is not None:
                    seq_info["motion_predictions"] = motion_predictions
            
            # 3. forward with motion-aware model:
            seq_info = self.model(seq_info=seq_info, part="trajectory_modeling")
            id_logits, _, _ = self.model(seq_info=seq_info, part="id_decoder")
            
            # 4. get scores:
            id_logits = id_logits[0, 0, 0]
            if not self.use_sigmoid:
                id_scores = id_logits.softmax(dim=-1)
            else:
                id_scores = id_logits.sigmoid()
                
            # 5. Motion-aware assignment
            if hasattr(self, 'trajectory_velocities') and self.trajectory_velocities.shape[0] > 0:
                # Adjust scores based on motion consistency
                motion_consistency_scores = self._compute_motion_consistency(boxes)
                id_scores = id_scores * (1 - self.motion_weight) + motion_consistency_scores * self.motion_weight
            
            # 6. assign id labels:
            match self.assignment_protocol:
                case "hungarian": id_labels = self._hungarian_assignment(id_scores=id_scores)
                case "object-max": id_labels = self._object_max_assignment(id_scores=id_scores)
                case "id-max": id_labels = self._id_max_assignment(id_scores=id_scores)
                case _: raise NotImplementedError

            id_pred_labels = torch.tensor(id_labels, dtype=torch.int64, device=distributed_device())
            return id_pred_labels

    def _get_motion_predictions(self, current_boxes):
        """
        Get motion predictions using Kalman filter
        """
        if not hasattr(self.model, 'trajectory_modeling') or not hasattr(self.model.trajectory_modeling, 'kalman_filter'):
            return None
            
        if self.trajectory_boxes.shape[0] == 0 or self.trajectory_boxes.shape[1] == 0: # No tracks or no history
            return None
        
        last_pos = self.trajectory_boxes[-1] # (N_obj_tracked, 4)
        if self.trajectory_boxes.shape[0] > 1:
            prev_pos = self.trajectory_boxes[-2]
            velocities = last_pos - prev_pos
        else: # First frame for these tracks, or only one frame of history
            velocities = torch.zeros_like(last_pos)
        
        predicted_boxes_for_tracks = last_pos + velocities
        return predicted_boxes_for_tracks # Shape (N_obj_tracked, 4)

    def _compute_motion_consistency(self, current_boxes): # current_boxes are (N_current_det, 4)
        """
        Compute motion consistency scores for better ID assignment
        """
        if self.trajectory_boxes.shape[0] < 1 or self.trajectory_boxes.shape[1] == 0: # No tracks or no history to predict from
            num_existing_tracks = self.trajectory_id_labels.shape[1] if self.trajectory_id_labels.nelement() > 0 else 0
            motion_scores = torch.zeros((len(current_boxes), num_existing_tracks), device=current_boxes.device)
            newborn_scores = torch.ones((len(current_boxes), 1), device=current_boxes.device) * 0.1 # Default newborn score
            return torch.cat([motion_scores, newborn_scores], dim=-1)

        last_boxes = self.trajectory_boxes[-1]  # (N_tracks, 4)
        velocities = torch.zeros_like(last_boxes)
        if self.trajectory_boxes.shape[0] > 1:
            velocities = self.trajectory_boxes[-1] - self.trajectory_boxes[-2]
        expected_boxes = last_boxes + velocities # (N_tracks, 4) -> These are predicted positions of existing tracks
        
        distances = torch.cdist(current_boxes, expected_boxes, p=2) # Shape: (N_current_det, N_tracks)
        
        # Convert distances to similarity scores
        motion_scores = torch.exp(-distances / 0.1)  # Adjust temperature as needed. Shape: (N_current_det, N_tracks)
        
        # Normalize scores across tracks for each detection
        # Each row sums to 1 (or close, due to clamp) over existing tracks
        motion_scores = motion_scores / motion_scores.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        # Pad with newborn scores. This score is for the option of assigning a detection as a 'newborn'
        # It should be one score per detection for the "newborn" class.
        newborn_scores = torch.ones((len(current_boxes), 1), device=current_boxes.device) * 0.1 
        motion_scores = torch.cat([motion_scores, newborn_scores], dim=-1) # Shape: (N_current_det, N_tracks + 1)
        
        return motion_scores
    
    def get_track_results(self):
        return self.current_track_results

    def _get_activate_detections(self, detr_out: dict):
        logits = detr_out["pred_logits"][0]
        boxes = detr_out["pred_boxes"][0]
        output_embeds = detr_out["outputs"][0]
        scores = logits.sigmoid()
        scores, categories = torch.max(scores, dim=-1)
        area = boxes[:, 2] * self.bbox_unnorm[2] * boxes[:, 3] * self.bbox_unnorm[3]
        activate_indices = (scores > self.det_thresh) & (area > self.area_thresh)
        # Selecting:
        # logits = logits[activate_indices] # Logits not used later, so removing this line is fine
        boxes = boxes[activate_indices]
        output_embeds = output_embeds[activate_indices]
        scores = scores[activate_indices]
        categories = categories[activate_indices]
        return scores, categories, boxes, output_embeds

    def _get_id_pred_labels(self, boxes: torch.Tensor, output_embeds: torch.Tensor):
        if self.trajectory_features.shape[0] == 0:
            return self.num_id_vocabulary * torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device)
        else:
            # 1. prepare current infos:
            current_features = output_embeds[None, ...]     # (T, N, ...)
            current_boxes = boxes[None, ...]                # (T, N, 4)
            current_masks = torch.zeros((1, output_embeds.shape[0]), dtype=torch.bool, device=distributed_device())
            current_times = self.trajectory_times.shape[0] * torch.ones(
                (1, output_embeds.shape[0]), dtype=torch.int64, device=distributed_device(),
            )
            # 2. prepare seq_info:
            seq_info = {
                "trajectory_features": self.trajectory_features[None, None, ...],
                "trajectory_boxes": self.trajectory_boxes[None, None, ...],
                "trajectory_id_labels": self.trajectory_id_labels[None, None, ...],
                "trajectory_times": self.trajectory_times[None, None, ...],
                "trajectory_masks": self.trajectory_masks[None, None, ...],
                "unknown_features": current_features[None, None, ...],
                "unknown_boxes": current_boxes[None, None, ...],
                "unknown_masks": current_masks[None, None, ...],
                "unknown_times": current_times[None, None, ...],
            }
            # 3. forward:
            seq_info = self.model(seq_info=seq_info, part="trajectory_modeling")
            id_logits, _, _ = self.model(seq_info=seq_info, part="id_decoder")
            # 4. get scores:
            id_logits = id_logits[0, 0, 0]
            if not self.use_sigmoid:
                id_scores = id_logits.softmax(dim=-1)
            else:
                id_scores = id_logits.sigmoid()
            # 5. assign id labels:
            # Different assignment protocols:
            match self.assignment_protocol:
                case "hungarian": id_labels = self._hungarian_assignment(id_scores=id_scores)
                case "object-max": id_labels = self._object_max_assignment(id_scores=id_scores)
                case "id-max": id_labels = self._id_max_assignment(id_scores=id_scores)
                # case "object-priority": id_labels = self._object_priority_assignment(id_scores=id_scores)
                case _: raise NotImplementedError

            id_pred_labels = torch.tensor(id_labels, dtype=torch.int64, device=distributed_device())
            return id_pred_labels

    def _assign_newborn_id_labels(self, pred_id_labels: torch.Tensor):
        # 1. how many newborn instances?
        n_newborns = (pred_id_labels == self.num_id_vocabulary).sum().item()
        if n_newborns == 0:
            return pred_id_labels
        else:
            # 2. get available id labels from id_queue:
            newborn_id_labels = torch.tensor(
                list(self.id_queue)[:n_newborns], dtype=torch.int64, device=distributed_device(),
            )
            # 3. make sure these id labels are not in trajectory infos:
            trajectory_remove_idxs = torch.zeros(
                self.trajectory_id_labels.shape[1] if self.trajectory_id_labels.nelement() > 0 else 0, 
                dtype=torch.bool, device=distributed_device(),
            )
            if self.trajectory_id_labels.shape[0] > 0 and self.trajectory_id_labels.shape[1] > 0 : # Check if trajectories exist
                for _ in range(len(newborn_id_labels)):
                    trajectory_remove_idxs |= (self.trajectory_id_labels[0] == newborn_id_labels[_])
                    if newborn_id_labels[_].item() in self.id_label_to_id:
                        self.id_label_to_id.pop(newborn_id_labels[_].item())
            else: # No existing trajectories, no removal needed, but still handle id_label_to_id
                 for _ in range(len(newborn_id_labels)):
                    if newborn_id_labels[_].item() in self.id_label_to_id: # Should not happen if queue is managed well
                        self.id_label_to_id.pop(newborn_id_labels[_].item())

            # remove from trajectory infos:
            if self.trajectory_id_labels.shape[1] > 0 : # Only if there are trajectories to remove from
                self.trajectory_features = self.trajectory_features[:, ~trajectory_remove_idxs]
                self.trajectory_boxes = self.trajectory_boxes[:, ~trajectory_remove_idxs]
                self.trajectory_id_labels = self.trajectory_id_labels[:, ~trajectory_remove_idxs]
                self.trajectory_times = self.trajectory_times[:, ~trajectory_remove_idxs]
                self.trajectory_masks = self.trajectory_masks[:, ~trajectory_remove_idxs]
                 # Also remove from motion fields if using motion
                if self.use_motion:
                    if hasattr(self, 'trajectory_velocities') and self.trajectory_velocities.shape[1] > 0:
                        self.trajectory_velocities = self.trajectory_velocities[:, ~trajectory_remove_idxs]
                    if hasattr(self, 'trajectory_accelerations') and self.trajectory_accelerations.shape[1] > 0:
                        self.trajectory_accelerations = self.trajectory_accelerations[:, ~trajectory_remove_idxs]

            # 4. assign id labels to newborn instances:
            pred_id_labels[pred_id_labels == self.num_id_vocabulary] = newborn_id_labels
            # 5. update id infos:
            for _ in range(len(newborn_id_labels)):
                self.id_label_to_id[newborn_id_labels[_].item()] = self.next_id
                self.id_queue.discard(newborn_id_labels[_].item()) # Remove from available pool
                self.next_id += 1

            return pred_id_labels

    def _update_trajectory_infos(self, boxes: torch.Tensor, output_embeds: torch.Tensor, id_labels: torch.Tensor):
        """
        Extended to update motion information
        """
        # 1. cut trajectory infos:
        self.trajectory_features = self.trajectory_features[-self.miss_tolerance + 2:, ...]
        self.trajectory_boxes = self.trajectory_boxes[-self.miss_tolerance + 2:, ...]
        self.trajectory_id_labels = self.trajectory_id_labels[-self.miss_tolerance + 2:, ...]
        self.trajectory_times = self.trajectory_times[-self.miss_tolerance + 2:, ...]
        self.trajectory_masks = self.trajectory_masks[-self.miss_tolerance + 2:, ...]
        
        # Also cut motion fields if using motion
        if self.use_motion:
            if hasattr(self, 'trajectory_velocities'):
                self.trajectory_velocities = self.trajectory_velocities[-self.miss_tolerance + 2:, ...]
            if hasattr(self, 'trajectory_accelerations'):
                self.trajectory_accelerations = self.trajectory_accelerations[-self.miss_tolerance + 2:, ...]
        
        # 2. find out all new instances:
        already_id_labels = set(self.trajectory_id_labels[0].tolist() if self.trajectory_id_labels.shape[0] > 0 and self.trajectory_id_labels.shape[1] > 0 else [])
        _id_labels = set(id_labels.tolist())
        newborn_id_labels_set = _id_labels - already_id_labels
        
        # 3. add newborn instances to trajectory infos:
        if len(newborn_id_labels_set) > 0:
            newborn_id_labels_tensor = torch.tensor(list(newborn_id_labels_set), dtype=torch.int64, device=distributed_device())
            _T = self.trajectory_id_labels.shape[0] # Current length of history (after cutting)
            _N_new = len(newborn_id_labels_tensor)
            
            # Create history for new tracks, padded with zeros/masks
            # If _T is 0 (no prior tracks or all cut), these will be (0, N_new, ...)
            _new_id_labels_hist = einops.repeat(newborn_id_labels_tensor, 'n -> t n', t=_T)
            _new_boxes_hist = torch.zeros((_T, _N_new, 4), dtype=self.dtype, device=distributed_device())
            _new_times_hist = einops.repeat(
                torch.arange(_T, dtype=torch.int64, device=distributed_device()), 't -> t n', n=_N_new,
            ) if _T > 0 else torch.zeros((0,_N_new), dtype=torch.int64, device=distributed_device()) # Handle _T=0
            _new_features_hist = torch.zeros(
                (_T, _N_new, 256), dtype=self.dtype, device=distributed_device(),
            )
            _new_masks_hist = torch.ones((_T, _N_new), dtype=torch.bool, device=distributed_device()) # Masked for past steps
            
            # Concatenate new tracks to existing trajectory tensors
            if self.trajectory_id_labels.shape[1] > 0 : # If there are existing tracks
                self.trajectory_id_labels = torch.cat([self.trajectory_id_labels, _new_id_labels_hist], dim=1)
                self.trajectory_boxes = torch.cat([self.trajectory_boxes, _new_boxes_hist], dim=1)
                self.trajectory_times = torch.cat([self.trajectory_times, _new_times_hist], dim=1)
                self.trajectory_features = torch.cat([self.trajectory_features, _new_features_hist], dim=1)
                self.trajectory_masks = torch.cat([self.trajectory_masks, _new_masks_hist], dim=1)
            else: # First tracks being added
                self.trajectory_id_labels = _new_id_labels_hist
                self.trajectory_boxes = _new_boxes_hist
                self.trajectory_times = _new_times_hist
                self.trajectory_features = _new_features_hist
                self.trajectory_masks = _new_masks_hist

            if self.use_motion:
                _new_velocities_hist = torch.zeros((_T, _N_new, 4), dtype=self.dtype, device=distributed_device())
                _new_accelerations_hist = torch.zeros((_T, _N_new, 4), dtype=self.dtype, device=distributed_device())
                if hasattr(self, 'trajectory_velocities') and self.trajectory_velocities.shape[1] > 0:
                    self.trajectory_velocities = torch.cat([self.trajectory_velocities, _new_velocities_hist], dim=1)
                    self.trajectory_accelerations = torch.cat([self.trajectory_accelerations, _new_accelerations_hist], dim=1)
                else: 
                    current_T_dim = self.trajectory_boxes.shape[0]
                    current_N_dim = self.trajectory_boxes.shape[1]
                    if not hasattr(self, 'trajectory_velocities') or self.trajectory_velocities.shape[1] == 0 :
                        self.trajectory_velocities = torch.zeros((current_T_dim, current_N_dim, 4), dtype=self.dtype, device=distributed_device())
                        self.trajectory_accelerations = torch.zeros((current_T_dim, current_N_dim, 4), dtype=self.dtype, device=distributed_device())
                    else: 
                         self.trajectory_velocities = torch.cat([self.trajectory_velocities, _new_velocities_hist], dim=1)
                         self.trajectory_accelerations = torch.cat([self.trajectory_accelerations, _new_accelerations_hist], dim=1)


        _N_total_tracks = self.trajectory_id_labels.shape[1] if self.trajectory_id_labels.nelement() > 0 else 0
        
        if self.trajectory_id_labels.shape[0] > 0 and _N_total_tracks > 0:
            current_id_labels_for_new_frame = self.trajectory_id_labels[-1].clone() 
        elif _N_total_tracks > 0: 
            current_id_labels_for_new_frame = newborn_id_labels_tensor.clone()
        else: 
            current_id_labels_for_new_frame = torch.empty((0,), dtype=torch.int64, device=distributed_device())


        current_features_for_new_frame = torch.zeros((_N_total_tracks, 256), dtype=self.dtype, device=distributed_device())
        current_boxes_for_new_frame = torch.zeros((_N_total_tracks, 4), dtype=self.dtype, device=distributed_device())
        new_time_step_val = self.trajectory_times.shape[0] 
        current_times_for_new_frame = new_time_step_val * torch.ones((_N_total_tracks,), dtype=torch.int64, device=distributed_device())
        current_masks_for_new_frame = torch.ones((_N_total_tracks,), dtype=torch.bool, device=distributed_device()) # Assume all masked initially for this new frame


        if _N_total_tracks > 0 and id_labels.numel() > 0 : 
            indices = torch.eq(current_id_labels_for_new_frame[:, None], id_labels[None, :]).nonzero(as_tuple=False)
            
            if indices.numel() > 0:
                track_indices_to_update = indices[:, 0] 
                detection_indices = indices[:, 1]       
            
                current_id_labels_for_new_frame[track_indices_to_update] = id_labels[detection_indices] 
                current_features_for_new_frame[track_indices_to_update] = output_embeds[detection_indices]
                current_boxes_for_new_frame[track_indices_to_update] = boxes[detection_indices]
                current_masks_for_new_frame[track_indices_to_update] = False 

        if self.trajectory_features.nelement() == 0 and _N_total_tracks > 0 : 
            self.trajectory_features = current_features_for_new_frame[None, ...]
            self.trajectory_boxes = current_boxes_for_new_frame[None, ...]
            self.trajectory_id_labels = current_id_labels_for_new_frame[None, ...]
            self.trajectory_times = current_times_for_new_frame[None, ...]
            self.trajectory_masks = current_masks_for_new_frame[None, ...]
        elif _N_total_tracks > 0 : 
            self.trajectory_features = torch.cat([self.trajectory_features, current_features_for_new_frame[None, ...]], dim=0).contiguous()
            self.trajectory_boxes = torch.cat([self.trajectory_boxes, current_boxes_for_new_frame[None, ...]], dim=0).contiguous()
            self.trajectory_id_labels = torch.cat([self.trajectory_id_labels, current_id_labels_for_new_frame[None, ...]], dim=0).contiguous()
            self.trajectory_times = torch.cat([self.trajectory_times, current_times_for_new_frame[None, ...]], dim=0).contiguous()
            self.trajectory_masks = torch.cat([self.trajectory_masks, current_masks_for_new_frame[None, ...]], dim=0).contiguous()

        if self.use_motion and self.trajectory_boxes.shape[0] > 0 and self.trajectory_boxes.shape[1] > 0: 
            if not hasattr(self, 'trajectory_velocities') or \
               self.trajectory_velocities.shape[1] != self.trajectory_boxes.shape[1] or \
               self.trajectory_velocities.shape[0] != self.trajectory_boxes.shape[0]-1: 
                
                if not hasattr(self, 'trajectory_velocities') or self.trajectory_velocities.shape[1] != self.trajectory_boxes.shape[1]:
                    self.trajectory_velocities = torch.zeros_like(self.trajectory_boxes)
                    self.trajectory_accelerations = torch.zeros_like(self.trajectory_boxes)
                    if self.trajectory_boxes.shape[0] > 1:
                        for t_idx in range(1, self.trajectory_boxes.shape[0]):
                            valid_prev = ~self.trajectory_masks[t_idx-1]
                            valid_curr = ~self.trajectory_masks[t_idx]
                            valid_both = valid_prev & valid_curr
                            if valid_both.any():
                                self.trajectory_velocities[t_idx, valid_both] = (self.trajectory_boxes[t_idx, valid_both] - self.trajectory_boxes[t_idx-1, valid_both])
                        if self.trajectory_boxes.shape[0] > 2: 
                             for t_idx in range(2, self.trajectory_boxes.shape[0]):
                                valid_vel_curr = ~self.trajectory_masks[t_idx] 
                                valid_vel_prev = ~self.trajectory_masks[t_idx-1]
                                valid_both_vel = valid_vel_curr & valid_vel_prev # If both points for vel_curr are valid, and both for vel_prev
                                if valid_both_vel.any():
                                     self.trajectory_accelerations[t_idx, valid_both_vel] = (self.trajectory_velocities[t_idx, valid_both_vel] - self.trajectory_velocities[t_idx-1, valid_both_vel])


            current_velocities_for_new_frame = torch.zeros((_N_total_tracks, 4), dtype=self.dtype, device=distributed_device())
            if self.trajectory_boxes.shape[0] > 1: 
                valid_prev = ~self.trajectory_masks[-2]
                valid_curr = ~self.trajectory_masks[-1] 
                valid_both = valid_prev & valid_curr
                if valid_both.any():
                    current_velocities_for_new_frame[valid_both] = (self.trajectory_boxes[-1][valid_both] - \
                                                                  self.trajectory_boxes[-2][valid_both])
            
            current_accelerations_for_new_frame = torch.zeros((_N_total_tracks, 4), dtype=self.dtype, device=distributed_device())
            if self.trajectory_velocities.shape[0] > 0 and self.trajectory_boxes.shape[0] > 1: 
                valid_vel_prev_frame = ~self.trajectory_masks[-2] 
                valid_vel_curr_frame = ~self.trajectory_masks[-1] 
                valid_for_accel = valid_vel_prev_frame & valid_vel_curr_frame 
                if valid_for_accel.any():
                    prev_velocities = self.trajectory_velocities[-1, valid_for_accel] 
                    curr_velocities_subset = current_velocities_for_new_frame[valid_for_accel]
                    current_accelerations_for_new_frame[valid_for_accel] = curr_velocities_subset - prev_velocities

            if self.trajectory_velocities.shape[0] == self.trajectory_boxes.shape[0]:
                self.trajectory_velocities[-1] = current_velocities_for_new_frame
                self.trajectory_accelerations[-1] = current_accelerations_for_new_frame
            else: # Append if history was shorter (e.g. T-1)
                self.trajectory_velocities = torch.cat([self.trajectory_velocities, current_velocities_for_new_frame[None, ...]], dim=0).contiguous()
                self.trajectory_accelerations = torch.cat([self.trajectory_accelerations, current_accelerations_for_new_frame[None, ...]], dim=0).contiguous()
 
        # 4.5. a hack implementation to fix "times":
        if self.trajectory_times.nelement() > 0 :
            self.trajectory_times = einops.repeat(
                torch.arange(self.trajectory_times.shape[0], dtype=torch.int64, device=distributed_device()),
                't -> t n', n=self.trajectory_times.shape[1],
            ).contiguous().clone()
        return

    def _filter_out_inactive_tracks(self):
        """
        Extended to also filter motion fields
        """
        if self.trajectory_masks.shape[0] == 0 or self.trajectory_masks.shape[1] == 0 : # No tracks or no history
            return

        is_active = torch.sum((~self.trajectory_masks).to(torch.int64), dim=0) > 0
        self.trajectory_features = self.trajectory_features[:, is_active]
        self.trajectory_boxes = self.trajectory_boxes[:, is_active]
        self.trajectory_id_labels = self.trajectory_id_labels[:, is_active]
        self.trajectory_times = self.trajectory_times[:, is_active]
        self.trajectory_masks = self.trajectory_masks[:, is_active]
        
        # Also filter motion fields if using motion
        if self.use_motion:
            if hasattr(self, 'trajectory_velocities') and self.trajectory_velocities.shape[1] > 0 :
                self.trajectory_velocities = self.trajectory_velocities[:, is_active]
            if hasattr(self, 'trajectory_accelerations') and self.trajectory_accelerations.shape[1] > 0:
                self.trajectory_accelerations = self.trajectory_accelerations[:, is_active]
        return

    def _hungarian_assignment(self, id_scores: torch.Tensor):
        # Keep original implementation
        id_labels = list()  # final ID labels
        if id_scores.numel() == 0: # No detections to assign
            return id_labels
        
        # id_scores are (N_det, N_classes) where N_classes = N_tracked_IDs + 1 (newborn)
        # N_tracked_IDs can be found from self.trajectory_id_labels[0] if it exists
        # Or, more robustly, id_scores.shape[1] - 1 is the number of active ID classes.
        
        num_detections = id_scores.shape[0]
        num_id_classes_with_newborn = id_scores.shape[1]
        
        if num_id_classes_with_newborn == 0 : # Should not happen if id_scores is not empty
             return [self.num_id_vocabulary] * num_detections
        
        active_track_ids_set = set()
        if self.trajectory_id_labels.shape[0] > 0 and self.trajectory_id_labels.shape[1] > 0:
            active_track_ids_set = set(self.trajectory_id_labels[-1].tolist()) # IDs active in last frame

        cost_matrix = (1 - id_scores).cpu() # cost = 1 - score
        
        # match_rows are detection indices, match_cols are class indices from id_scores
        match_rows, match_cols = linear_sum_assignment(cost_matrix.numpy()) 
        
        assigned_track_ids_in_current_step = set()
        # Initialize all detections as "newborn"
        id_labels = [self.num_id_vocabulary] * num_detections

        for i in range(len(match_rows)):
            det_idx = match_rows[i]
            id_class_idx = match_cols[i] # This is an index into the columns of id_scores

            # id_class_idx could be an existing track ID or the index for "newborn" class
            # If id_scores columns are [track_id_0, track_id_1, ..., track_id_k-1, newborn_class_idx]
            # then id_class_idx needs to be mapped to actual track ID or self.num_id_vocabulary

            # This assumes id_class_idx from hungarian IS the actual ID label if not newborn class
            # And that the last column of id_scores (index num_id_classes_with_newborn - 1) is "newborn"
            
            score_for_match = id_scores[det_idx, id_class_idx]
            is_newborn_class_match = (id_class_idx == num_id_classes_with_newborn -1) # Assuming last class is newborn type

            potential_id_label = id_class_idx # If not newborn class, this is the track ID from model's vocab
                                              # This needs to be an actual track ID from *active_track_ids_set*
                                              # The original code: `_id = match_cols[_]` then checks `_id not in trajectory_id_labels_set`
                                              # This implies match_cols[_] *are* the id_labels from the vocabulary.

            _id = id_class_idx # This is the ID label (0 to num_id_vocabulary-1) or num_id_vocabulary for newborn virtual class

            if _id == self.num_id_vocabulary : # Explicitly matched to newborn class by Hungarian
                 id_labels[det_idx] = self.num_id_vocabulary
            elif _id not in active_track_ids_set: # Matched to an ID that is not currently active (e.g. from vocab but not tracked)
                id_labels[det_idx] = self.num_id_vocabulary
            elif score_for_match < self.id_thresh: # Matched to an active track, but score too low
                id_labels[det_idx] = self.num_id_vocabulary
            elif _id in assigned_track_ids_in_current_step: # Track ID already assigned to another detection (higher score by Hungarian)
                id_labels[det_idx] = self.num_id_vocabulary # This detection becomes newborn
            else: # Good match
                id_labels[det_idx] = _id
                assigned_track_ids_in_current_step.add(_id)
        return id_labels

    def _object_max_assignment(self, id_scores: torch.Tensor):
        id_labels = list()  # final ID labels
        if id_scores.numel() == 0: return id_labels
        
        active_trajectory_id_labels_set = set()
        if self.trajectory_id_labels.shape[0] > 0 and self.trajectory_id_labels.shape[1] > 0:
             active_trajectory_id_labels_set = set(self.trajectory_id_labels[-1].tolist())   # all tracked ID labels

        object_max_confs, object_assigned_id_labels = torch.max(id_scores, dim=-1)   # get the target ID labels and confs for each object
        
        # Get the max confs for each ID label across all objects that prefer it
        id_max_confs_for_each_label = {} # Store the highest confidence an object has for a given ID label
        for det_idx in range(len(object_assigned_id_labels)):
            _id_label = object_assigned_id_labels[det_idx].item()
            _conf = object_max_confs[det_idx].item()
            if _id_label == self.num_id_vocabulary: # Skip newborn class for this dict
                continue
            if _id_label not in id_max_confs_for_each_label:
                id_max_confs_for_each_label[_id_label] = _conf
            else:
                id_max_confs_for_each_label[_id_label] = max(id_max_confs_for_each_label[_id_label], _conf)
        
        # Assign ID labels
        assigned_ids_in_this_step = set()
        for det_idx in range(len(object_assigned_id_labels)):
            _id_label_for_object = object_assigned_id_labels[det_idx].item() # ID this object wants most
            _conf_for_object = object_max_confs[det_idx].item()

            if _id_label_for_object == self.num_id_vocabulary: # Object prefers to be newborn
                id_labels.append(self.num_id_vocabulary)
            elif _id_label_for_object not in active_trajectory_id_labels_set: # Object wants an ID not currently active
                id_labels.append(self.num_id_vocabulary)
            elif _conf_for_object < self.id_thresh: # Confidence too low
                id_labels.append(self.num_id_vocabulary)
            elif _conf_for_object < id_max_confs_for_each_label.get(_id_label_for_object, -1.0): # Not the best object for this ID
                id_labels.append(self.num_id_vocabulary)
            elif _id_label_for_object in assigned_ids_in_this_step: # This ID was already taken by a better object
                id_labels.append(self.num_id_vocabulary)
            else: # Assign
                id_labels.append(_id_label_for_object)
                assigned_ids_in_this_step.add(_id_label_for_object)
        return id_labels

    def _id_max_assignment(self, id_scores: torch.Tensor):
        # id_scores: (N_det, N_classes) where N_classes include active IDs and newborn
        # Initialize all detections to newborn
        id_labels = [self.num_id_vocabulary] * id_scores.shape[0]
        if id_scores.numel() == 0: return id_labels

        active_trajectory_id_labels_set = set()
        if self.trajectory_id_labels.shape[0] > 0 and self.trajectory_id_labels.shape[1] > 0:
             active_trajectory_id_labels_set = set(self.trajectory_id_labels[-1].tolist()) 

        # For each ID class (column in id_scores), find the detection (row) that has max score for it.
        # id_max_confs_for_class: max score for each class
        # id_max_obj_idxs_for_class: index of detection that gave that max score
        id_max_confs_for_class, id_max_obj_idxs_for_class = torch.max(id_scores, dim=0)

        # For each detection, find what is the max score it could have gotten if it were chosen by an ID class.
        obj_max_conf_if_chosen = [-1.0] * id_scores.shape[0]
        for class_idx in range(id_scores.shape[1]):
            obj_idx = id_max_obj_idxs_for_class[class_idx].item()
            conf = id_max_confs_for_class[class_idx].item()
            obj_max_conf_if_chosen[obj_idx] = max(obj_max_conf_if_chosen[obj_idx], conf)

        # Assign ID labels
        assigned_detections = [False] * id_scores.shape[0]
        for class_idx in range(len(id_max_obj_idxs_for_class)):
            _id_label = class_idx # Assuming class_idx maps to the actual ID label or newborn index
                                  # This needs to align with how id_scores columns are structured.
                                  # Typically, class_idx would be the ID itself (0 to num_id_vocabulary-1)
                                  # or a specific index for the "newborn" class.

            if _id_label == self.num_id_vocabulary: # This class is the 'newborn' type, skip assignment here.
                continue
            
            _obj_idx_for_this_id = id_max_obj_idxs_for_class[_id_label].item() # Obj that wants this ID most
            _conf_this_id_gets = id_max_confs_for_class[_id_label].item()      # The score for that

            if assigned_detections[_obj_idx_for_this_id]: # This object already got an ID
                continue 
            if _conf_this_id_gets < self.id_thresh: # Score too low
                continue
            if _id_label not in active_trajectory_id_labels_set: # ID is not active
                continue
            # Check if this object (_obj_idx_for_this_id) is indeed the best recipient for _id_label,
            # and also if _id_label is the best choice for this object.
            # The original logic: _conf < object_max_confs[_obj_idx]
            # object_max_confs was based on object_max_id_labels from _object_max_assignment.
            # Here, object_max_confs is obj_max_conf_if_chosen.
            if _conf_this_id_gets < obj_max_conf_if_chosen[_obj_idx_for_this_id] - 1e-5: # This ID is not the best choice for this obj
                                                                                        # (allow for small float inaccuracies)
                continue
            
            id_labels[_obj_idx_for_this_id] = _id_label
            assigned_detections[_obj_idx_for_this_id] = True
            
        return id_labels
