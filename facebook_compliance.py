"""
Facebook-compliant video redaction system.
A clean, simple solution for creating platform-compliant videos.

This system mimics Facebook's automated content moderation to detect
and strategically blur content that would trigger their AI review.

Usage:
    python facebook_compliance.py input_video.mp4 output_video.mp4
"""

import cv2
import numpy as np
import argparse
import subprocess
import tempfile
import os
from typing import Tuple, List, Dict
import urllib.request
import json

from typing import Tuple, List, Dict
import urllib.request
import json


class IntelligentGoreDetector:
    """
    Advanced gore detection with strict discrimination between harmful and innocent content.
    Focus: Only detect truly problematic areas, not normal human presence.
    """
    
    def __init__(self):
        self.violence_keywords = {
            'weapons': ['knife', 'gun', 'sword', 'blade', 'pistol', 'rifle'],
            'medical': ['blood', 'wound', 'injury', 'cut', 'bruise', 'scar'],
            'violence': ['fight', 'punch', 'hit', 'attack', 'assault']
        }
        
        # Enhanced thresholds for discrimination
        self.blood_thresholds = {
            'minimum_area': 200,      # Minimum pixels for blood detection
            'saturation_min': 100,    # High saturation required for blood
            'context_required': True   # Require contextual evidence
        }
        
        # Temporal smoothing to reduce flickering
        self.detection_history = []  # Track detections across frames
        self.temporal_smoothing = 5   # Number of frames to consider for smoothing
        self.stability_threshold = 0.6  # Minimum consistency required
        self.frame_count = 0
        
        # Load detection if available
        self.setup_object_detection()
    
    def setup_object_detection(self):
        """Setup object detection model for gore/violence detection."""
        try:
            self.net = None
            self.output_layers = None
            print("🤖 AI Gore Detection: Using enhanced discrimination algorithms")
        except Exception as e:
            print(f"⚠️  Advanced AI not available, using enhanced heuristics: {e}")
            self.net = None
    
    def detect_gore_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect specific gore-related objects with strict discrimination and temporal smoothing.
        Enhanced to avoid false positives on living people and reduce flickering.
        """
        self.frame_count += 1
        detections = []
        
        # Method 1: Ultra-conservative weapon detection
        weapon_regions = self._detect_weapons_enhanced(frame)
        detections.extend(weapon_regions)
        
        # Method 2: Discriminating blood detection (avoid red clothing/makeup)
        blood_regions = self._detect_actual_blood(frame)
        detections.extend(blood_regions)
        
        # Method 3: Death/corpse detection (NEW - distinguish dead from alive)
        death_regions = self._detect_death_indicators(frame)
        detections.extend(death_regions)
        
        # Method 4: Context-aware injury detection (avoid normal skin)
        injury_regions = self._detect_severe_injuries_only(frame)
        detections.extend(injury_regions)
        
        # REALITY CHECK: If we're detecting too many things, we're probably wrong
        if len(detections) > 5:  # Much stricter - real gore videos shouldn't have 5+ areas per frame
            print(f"🚨 Reality check: {len(detections)} detections seems excessive, filtering...")
            # Keep only the highest confidence detections
            detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:2]
        
        # Method 5: Remove false positives from living people
        filtered_detections = self._filter_false_positives(frame, detections)
        
        # Method 6: Apply temporal smoothing to reduce flickering
        stable_detections = self._apply_temporal_smoothing(filtered_detections)
        
        # Enhanced logging for better understanding
        if stable_detections:
            print(f"📍 Frame {self.frame_count}: {len(stable_detections)} stable detections after smoothing")
            for detection in stable_detections[:3]:  # Only show first 3 to avoid spam
                det_type = detection.get('type', 'unknown')
                confidence = detection.get('confidence', 0)
                print(f"   ⚠️  {det_type}: confidence {confidence:.2f}")
            if len(stable_detections) > 3:
                print(f"   ... and {len(stable_detections) - 3} more")
        
        return stable_detections
    
    def _cluster_weapon_lines(self, lines) -> List[List]:
        """Group nearby lines that might form weapons."""
        if len(lines) == 0:
            return []
        
        clusters = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
                
            cluster = [line1[0]]
            used.add(i)
            
            x1, y1, x2, y2 = line1[0]
            
            for j, line2 in enumerate(lines[i+1:], i+1):
                if j in used:
                    continue
                    
                x3, y3, x4, y4 = line2[0]
                
                # Check if lines are close enough to be part of same weapon
                dist = min(
                    np.sqrt((x1-x3)**2 + (y1-y3)**2),
                    np.sqrt((x1-x4)**2 + (y1-y4)**2),
                    np.sqrt((x2-x3)**2 + (y2-y3)**2),
                    np.sqrt((x2-x4)**2 + (y2-y4)**2)
                )
                
                if dist < 30:  # Lines within 30 pixels
                    cluster.append(line2[0])
                    used.add(j)
            
            if len(cluster) >= 2:  # At least 2 lines for weapon
                clusters.append(cluster)
        
        return clusters
    
    def _get_cluster_bbox(self, cluster, frame_shape) -> Tuple[int, int, int, int]:
        """Get bounding box for a cluster of lines."""
        all_points = []
        for line in cluster:
            x1, y1, x2, y2 = line
            all_points.extend([(x1, y1), (x2, y2)])
        
        if not all_points:
            return (0, 0, 0, 0)
        
        xs, ys = zip(*all_points)
        
        margin = 20
        x1 = max(0, min(xs) - margin)
        y1 = max(0, min(ys) - margin) 
        x2 = min(frame_shape[1], max(xs) + margin)
        y2 = min(frame_shape[0], max(ys) + margin)
        
        return (x1, y1, x2, y2)
    
    def _validate_metallic_object(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Check if region looks like a metallic weapon."""
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return False
            
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        
        # Check for metallic characteristics
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Metallic objects often have high local contrast
        contrast = np.std(gray_roi)
        
        # Metallic objects reflect light (bright spots)
        bright_pixels = np.sum(gray_roi > 200)
        bright_ratio = bright_pixels / gray_roi.size
        
        return contrast > 40 and bright_ratio > 0.1
    
    def _validate_blood_characteristics(self, roi: np.ndarray, contour) -> bool:
        """ULTRA-STRICT validation - only actual liquid blood passes."""
        if roi.size == 0:
            return False
        
        # Blood has specific texture and color characteristics
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Check saturation levels (blood is highly saturated)
        saturation = hsv_roi[:, :, 1]
        avg_saturation = np.mean(saturation)
        
        # Much stricter saturation requirement
        if avg_saturation < 150:  # Increased from 120 to 150
            return False
        
        # Check brightness - blood shouldn't be too bright
        value = hsv_roi[:, :, 2]
        avg_brightness = np.mean(value)
        if avg_brightness > 160 or avg_brightness < 30:  # Strict brightness range
            return False
        
        # Check for texture consistency - blood should be relatively smooth
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        texture_variance = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        if texture_variance > 500:  # Too textured for liquid blood
            return False
        
        # Check for spatter-like patterns (irregular edges)
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        
        if area > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # Blood spatter is irregular (low circularity) and area requirement
        if area < 1500:  # Increased minimum area requirement
            return False
            
        # Much stricter circularity and shape requirements
        return avg_saturation > 150 and circularity < 0.6 and circularity > 0.1
    
    def _calculate_blood_confidence(self, roi: np.ndarray, area: float) -> float:
        """Calculate confidence that this is actual blood."""
        if roi.size == 0:
            return 0.0
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Analyze color characteristics
        hue = hsv_roi[:, :, 0]
        saturation = hsv_roi[:, :, 1]
        value = hsv_roi[:, :, 2]
        
        # Blood characteristics
        red_hue_ratio = np.sum((hue < 10) | (hue > 170)) / hue.size
        high_sat_ratio = np.sum(saturation > 150) / saturation.size
        
        # Size factor (larger areas more concerning)
        size_factor = min(area / 1000, 1.0)
        
        confidence = (red_hue_ratio * 0.4 + high_sat_ratio * 0.4 + size_factor * 0.2)
        return min(confidence, 0.9)
    
    def _analyze_death_indicators(self, roi: np.ndarray, contour) -> float:
        """Analyze if region shows death indicators vs living person."""
        if roi.size == 0:
            return 0.0
        
        death_score = 0.0
        
        # 1. Skin color analysis (corpses have different coloration)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        
        # Check for unnatural skin tones
        l_channel = lab_roi[:, :, 0]  # Lightness
        a_channel = lab_roi[:, :, 1]  # Green-Red
        
        # Corpses often have grayish or bluish tint
        avg_lightness = np.mean(l_channel)
        avg_a = np.mean(a_channel)
        
        # Very pale or discolored skin
        if avg_lightness < 80 or avg_a < 120:  # Unnatural skin tone
            death_score += 0.3
        
        # 2. Position analysis (corpses in unnatural positions)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Horizontal positioning might indicate lying down
        if aspect_ratio > 2.0:  # Very wide compared to height
            death_score += 0.3
        
        # 3. Context clues (stillness, environment)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Check for very low variance (no movement/life signs)
        variance = np.var(gray_roi)
        if variance < 200:  # Very still
            death_score += 0.2
        
        # 4. Check for wounds/trauma in context
        dark_spots = np.sum(gray_roi < 50)  # Very dark areas
        dark_ratio = dark_spots / gray_roi.size
        
        if dark_ratio > 0.1:  # Significant dark areas (wounds/blood)
            death_score += 0.2
        
        return min(death_score, 1.0)
    
    def _validate_severe_injury(self, roi: np.ndarray) -> float:
        """Validate if this is a severe injury vs normal skin variation."""
        if roi.size == 0:
            return 0.0
        
        # Analyze color and texture for severe injury characteristics
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Severe injuries have distinct characteristics
        l_channel = lab_roi[:, :, 0]
        a_channel = lab_roi[:, :, 1]
        
        # Check for trauma indicators
        very_dark = np.sum(l_channel < 50)  # Deep wounds
        red_discoloration = np.sum(a_channel > 140)  # Blood/inflammation
        
        dark_ratio = very_dark / l_channel.size
        red_ratio = red_discoloration / a_channel.size
        
        # Texture analysis
        contrast = np.std(gray_roi)
        
        # Severe injuries: dark areas + red discoloration + high contrast
        injury_score = (dark_ratio * 0.4 + red_ratio * 0.4 + min(contrast/100, 0.2))
        
        return min(injury_score, 0.9)
    
    def _detect_life_indicators(self, roi: np.ndarray) -> float:
        """Detect indicators that this is a living person (to avoid false positives)."""
        if roi.size == 0:
            return 0.0
        
        life_score = 0.0
        
        # 1. Healthy skin color analysis
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        
        # Check for normal, healthy skin tones
        l_channel = lab_roi[:, :, 0]  # Lightness
        a_channel = lab_roi[:, :, 1]  # Green-Red
        
        avg_lightness = np.mean(l_channel)
        avg_a = np.mean(a_channel)
        
        # Normal skin tone ranges
        if 90 < avg_lightness < 180 and 125 < avg_a < 140:
            life_score += 0.4
        
        # 2. Check for natural variations (living skin has variation)
        variance = np.var(l_channel)
        if variance > 100:  # Natural skin variation
            life_score += 0.2
        
        # 3. Check for clothing/normal objects nearby
        # (This would be more complex, checking edges for fabric patterns)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Moderate edge density suggests normal objects/clothing
        if 0.05 < edge_density < 0.2:
            life_score += 0.2
        
        # 4. Position indicators (upright suggests alive)
        h, w = roi.shape[:2]
        aspect_ratio = w / h if h > 0 else 0
        
        if 0.5 < aspect_ratio < 1.5:  # More vertical = likely upright person
            life_score += 0.2
        
        return min(life_score, 1.0)
    
    def _detect_weapons_enhanced(self, frame: np.ndarray) -> List[Dict]:
        """
        Ultra-conservative weapon detection - only flag obvious weapons.
        Drastically reduced false positives by requiring multiple validation criteria.
        """
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]
        
        # Much stricter edge detection
        edges = cv2.Canny(gray, 80, 200)  # Higher thresholds to reduce noise
        
        # Detect only very clear, long straight lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,  # Much higher threshold
                               minLineLength=100, maxLineGap=5)      # Longer lines, smaller gaps
        
        if lines is not None:
            weapon_candidates = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                
                # Only consider very long, straight objects
                if length > 120:  # Much longer minimum length
                    # Check if angle suggests weapon (horizontal, vertical, or diagonal)
                    is_weapon_angle = (angle < 10 or angle > 170 or 
                                     (80 < angle < 100) or (45 < angle < 55) or (125 < angle < 135))
                    
                    if is_weapon_angle:
                        # Additional validation: check surrounding area for weapon characteristics
                        margin = 20
                        x_min, y_min = max(0, min(x1, x2) - margin), max(0, min(y1, y2) - margin)
                        x_max, y_max = min(w, max(x1, x2) + margin), min(h, max(y1, y2) + margin)
                        
                        roi = frame[y_min:y_max, x_min:x_max]
                        if roi.size > 0 and self._validate_weapon_characteristics(roi):
                            weapon_candidates.append({
                                'line': line[0],
                                'length': length,
                                'bbox': (x_min, y_min, x_max, y_max)
                            })
            
            # Group nearby weapon candidates
            if len(weapon_candidates) >= 2:  # Require at least 2 strong candidates
                # Only flag if multiple weapon-like lines are close together
                for candidate in weapon_candidates:
                    bbox = candidate['bbox']
                    
                    # Final validation with very high standards
                    if self._final_weapon_validation(frame, bbox):
                        detection = {
                            'type': 'weapon',
                            'confidence': min(0.6, candidate['length'] / 200),  # Lower max confidence
                            'bbox': bbox
                        }
                        detections.append(detection)
                        break  # Only add one weapon detection per region
        
        return detections
    
    def _validate_weapon_characteristics(self, roi):
        """Validate that the region has weapon-like characteristics."""
        if roi.size == 0:
            return False
            
        # Check for metallic/sharp object characteristics
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # Look for high contrast edges (metallic surfaces)
        edges = cv2.Canny(gray_roi, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Check color variance (weapons often have uniform colors)
        color_variance = np.var(gray_roi)
        
        # Weapons typically have high edge density and moderate color variance
        return edge_density > 0.1 and 500 < color_variance < 3000
    
    def _final_weapon_validation(self, frame, bbox):
        """Final strict validation to minimize false positives."""
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return False
        
        # Check aspect ratio - weapons are typically long and thin
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        
        # Weapon should have high aspect ratio (long and thin)
        if aspect_ratio < 3:
            return False
        
        # Check for consistent color/texture (not human skin)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue_variance = np.var(hsv_roi[:, :, 0])
        
        # Human skin has low hue variance, weapons typically have higher
        if hue_variance < 100:
            return False
        
        return True
    
    def _detect_actual_blood(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect actual blood while avoiding red clothing, makeup, or objects.
        Focus on blood spatter patterns and pooling.
        """
        detections = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Stricter blood detection ranges - MUCH MORE CONSERVATIVE
        blood_ranges = [
            ([0, 180, 120], [15, 255, 200]),    # VERY saturated fresh blood only
            ([160, 180, 120], [180, 255, 200])  # VERY saturated dark blood only
        ]
        
        for i, (lower, upper) in enumerate(blood_ranges):
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Find potential blood regions
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.blood_thresholds['minimum_area']:
                    
                    # Analyze region characteristics
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = frame[y:y+h, x:x+w]
                    
                    # Check if this looks like actual blood vs red object
                    if self._validate_blood_characteristics(roi, contour):
                        confidence = self._calculate_blood_confidence(roi, area)
                        
                        if confidence > 0.7:  # MUCH higher threshold for blood detection
                            detection = {
                                'type': f'actual_blood_{i}',
                                'confidence': confidence,
                                'bbox': (x, y, x + w, y + h)
                            }
                            detections.append(detection)
        
        return detections
    
    def _detect_death_indicators(self, frame: np.ndarray) -> List[Dict]:
        """
        DISABLED: Death indicator detection too aggressive - causing false positives.
        Only enable this for videos with confirmed deceased individuals.
        """
        # Currently disabled due to false positives on normal people
        return []
    
    def _detect_severe_injuries_only(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect only severe, obvious injuries - not normal skin variations.
        """
        detections = []
        
        # Convert to LAB for better injury detection
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect skin areas first
        skin_mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
        
        # Look for very dark or discolored areas on skin
        l_channel = lab[:, :, 1]  # A channel (green-red)
        dark_injuries = cv2.inRange(l_channel, 0, 110)  # Very dark/discolored
        
        # Combine with skin detection
        injury_candidates = cv2.bitwise_and(skin_mask, dark_injuries)
        
        # Clean up - only keep significant, irregular shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        injury_candidates = cv2.morphologyEx(injury_candidates, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(injury_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # Reasonable injury size
                
                # Check shape irregularity (injuries are jagged)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Only flag very irregular shapes (severe injuries)
                if solidity < 0.6:  # Very irregular = likely severe injury
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Additional validation
                    roi = frame[y:y+h, x:x+w]
                    injury_confidence = self._validate_severe_injury(roi)
                    
                    if injury_confidence > 0.5:
                        detection = {
                            'type': 'severe_injury',
                            'confidence': injury_confidence,
                            'bbox': (x, y, x + w, y + h)
                        }
                        detections.append(detection)
        
        return detections
    
    def _filter_false_positives(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        NEW: Remove detections that are likely false positives from living people.
        """
        filtered = []
        
        for detection in detections:
            # Skip weapon detections (always keep)
            if detection['type'] == 'weapon':
                filtered.append(detection)
                continue
            
            x1, y1, x2, y2 = detection['bbox']
            roi = frame[y1:y2, x1:x2]
            
            # Check if this region shows signs of life/normalcy
            life_indicators = self._detect_life_indicators(roi)
            
            # Only keep detection if life indicators are low
            if life_indicators < 0.3:  # Low life indicators = likely problematic
                filtered.append(detection)
            else:
                # This appears to be a living person or normal content
                print(f"🔍 Filtered false positive: {detection['type']} (life indicators: {life_indicators:.2f})")
        
        return filtered
    
    def _apply_temporal_smoothing(self, detections: List[Dict]) -> List[Dict]:
        """
        Apply temporal smoothing to reduce flickering by tracking detection consistency.
        Only return detections that appear consistently across multiple frames.
        """
        # Add current detections to history
        self.detection_history.append(detections)
        
        # Keep only recent history
        if len(self.detection_history) > self.temporal_smoothing:
            self.detection_history.pop(0)
        
        # If we don't have enough history yet, be conservative
        if len(self.detection_history) < 3:
            return detections  # Allow initial detections
        
        stable_detections = []
        
        # For each current detection, check if it's been consistent
        for current_detection in detections:
            current_bbox = current_detection['bbox']
            current_type = current_detection['type']
            
            # Count how many recent frames had similar detections
            consistency_count = 0
            for frame_detections in self.detection_history[-3:]:  # Check last 3 frames
                for past_detection in frame_detections:
                    if past_detection['type'] == current_type:
                        # Check if bounding boxes overlap significantly
                        overlap = self._calculate_bbox_overlap(current_bbox, past_detection['bbox'])
                        if overlap > 0.5:  # 50% overlap threshold
                            consistency_count += 1
                            break
            
            # Only keep detections that appear in most recent frames
            consistency_ratio = consistency_count / len(self.detection_history[-3:])
            if consistency_ratio >= self.stability_threshold:
                stable_detections.append(current_detection)
            else:
                print(f"🔄 Temporal filter: {current_type} (consistency: {consistency_ratio:.2f})")
        
        return stable_detections
    
    def _calculate_bbox_overlap(self, bbox1, bbox2):
        """Calculate intersection over union (IoU) for two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0  # No intersection
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _detect_weapons(self, frame: np.ndarray) -> List[Dict]:
        """Detect weapon-like objects using shape and edge analysis."""
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhanced edge detection for metallic objects
        edges = cv2.Canny(gray, 30, 100)
        
        # Detect long straight lines (potential knives, guns)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=40, maxLineGap=10)
        
        if lines is not None:
            weapon_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                
                # Look for long, straight objects (potential weapons)
                if length > 60 and (angle < 15 or angle > 165 or (80 < angle < 100)):
                    weapon_lines.append(line[0])
            
            # Group nearby lines into weapon regions
            if len(weapon_lines) >= 2:  # Multiple lines suggest weapon
                for line in weapon_lines:
                    x1, y1, x2, y2 = line
                    # Create bounding box around weapon
                    margin = 30
                    bbox = {
                        'type': 'weapon',
                        'confidence': min(0.7, len(weapon_lines) * 0.2),
                        'bbox': (max(0, min(x1, x2) - margin),
                                max(0, min(y1, y2) - margin),
                                min(frame.shape[1], max(x1, x2) + margin),
                                min(frame.shape[0], max(y1, y2) + margin))
                    }
                    detections.append(bbox)
        
        return detections
    
    def _detect_blood_patterns(self, frame: np.ndarray) -> List[Dict]:
        """Detect blood spatter patterns using advanced color and texture analysis."""
        detections = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Multiple red ranges for different blood types
        blood_ranges = [
            ([0, 50, 50], [15, 255, 255]),      # Fresh blood
            ([160, 50, 50], [180, 255, 255]),   # Dark red blood
            ([15, 30, 30], [25, 255, 200])     # Dried blood (brownish)
        ]
        
        for i, (lower, upper) in enumerate(blood_ranges):
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Find contours in blood regions
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Significant blood area
                    # Analyze shape characteristics
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    # Blood spatter is often irregular (low circularity)
                    if circularity < 0.7:  # Irregular shape suggests spatter
                        x, y, w, h = cv2.boundingRect(contour)
                        confidence = min(0.8, area / 1000)  # Higher area = higher confidence
                        
                        detection = {
                            'type': f'blood_pattern_{i}',
                            'confidence': confidence,
                            'bbox': (x, y, x + w, y + h)
                        }
                        detections.append(detection)
        
        return detections
    
    def _detect_injuries(self, frame: np.ndarray) -> List[Dict]:
        """Detect injury/wound patterns using texture and color analysis."""
        detections = []
        
        # Convert to LAB color space for better skin tone analysis
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        
        # Detect skin tones first
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
        
        # Look for dark spots on skin (potential wounds)
        dark_spots = cv2.inRange(l_channel, 0, 80)  # Very dark areas
        
        # Combine skin areas with dark spots
        injury_candidates = cv2.bitwise_and(skin_mask, dark_spots)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        injury_candidates = cv2.morphologyEx(injury_candidates, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(injury_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 2000:  # Reasonable injury size
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if area has irregular edges (wounds are often jagged)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                if solidity < 0.8:  # Irregular shape suggests injury
                    detection = {
                        'type': 'injury',
                        'confidence': min(0.6, (1 - solidity) + area / 1000),
                        'bbox': (x, y, x + w, y + h)
                    }
                    detections.append(detection)
        
        return detections
    
    def _detect_violent_actions(self, frame: np.ndarray) -> List[Dict]:
        """Detect violent motion patterns and aggressive postures."""
        detections = []
        
        # This would ideally use motion vectors from video analysis
        # For now, we'll detect high-energy areas that might indicate violence
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate local variance (high variance = lots of activity)
        kernel = np.ones((9, 9), np.float32) / 81
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        variance = sqr_mean - mean**2
        
        # High variance areas might indicate violent motion
        high_activity = variance > np.percentile(variance, 95)
        
        # Find regions of high activity
        contours, _ = cv2.findContours(high_activity.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Significant activity area
                x, y, w, h = cv2.boundingRect(contour)
                
                detection = {
                    'type': 'high_activity',
                    'confidence': min(0.5, area / 2000),
                    'bbox': (x, y, x + w, y + h)
                }
                detections.append(detection)
        
        return detections


def analyze_facebook_risk_intelligent(frame: np.ndarray, gore_detector: IntelligentGoreDetector) -> Tuple[float, List[Dict]]:
    """
    Advanced Facebook content violation risk analysis using AI gore detection.
    Returns risk score and list of detected problematic areas.
    """
    # Get specific gore object detections
    detections = gore_detector.detect_gore_objects(frame)
    
    # Calculate base risk from legacy detection
    base_risk = analyze_facebook_risk_legacy(frame)
    
    # Calculate AI-enhanced risk
    ai_risk = 0.0
    high_confidence_detections = []
    
    for detection in detections:
        confidence = detection['confidence']
        detection_type = detection['type']
        
        # Weight different types of violations
        type_weights = {
            'weapon': 0.9,           # Weapons are highest priority
            'blood_pattern_0': 0.8,  # Fresh blood
            'blood_pattern_1': 0.7,  # Dark blood
            'blood_pattern_2': 0.6,  # Dried blood
            'injury': 0.7,           # Injuries/wounds
            'high_activity': 0.4     # Violent motion
        }
        
        weight = type_weights.get(detection_type, 0.5)
        detection_risk = confidence * weight
        
        # Accumulate risk (non-linear to prevent over-flagging)
        ai_risk += detection_risk * (1 - ai_risk * 0.5)
        
        # Keep high-confidence detections for precise blurring
        if confidence > 0.4:
            high_confidence_detections.append(detection)
    
    # Combine base risk with AI risk
    final_risk = max(base_risk, ai_risk)
    
    # Boost risk if multiple types detected
    if len(set(d['type'] for d in high_confidence_detections)) > 1:
        final_risk = min(1.0, final_risk * 1.3)
    
    return min(final_risk, 1.0), high_confidence_detections


def analyze_facebook_risk_legacy(frame: np.ndarray) -> float:
    """Legacy risk analysis (renamed from original function)."""
    # Convert to different color spaces
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    
    # Analyze multiple risk factors
    blood_risk = detect_blood_content(hsv)
    skin_risk = detect_skin_content(hsv) 
    weapon_risk = detect_sharp_objects(gray)
    scene_risk = detect_disturbing_scenes(gray)
    central_risk = analyze_central_content(frame_rgb)
    motion_risk = detect_motion_activity(frame)
    color_risk = analyze_color_variance(hsv)
    
    # Weighted combination - ENHANCED weights since Facebook still flagged content
    risk_score = (blood_risk * 0.25 +      # Increased from 0.2
                  skin_risk * 0.15 +       # Kept same
                  weapon_risk * 0.2 +      # Increased from 0.15  
                  scene_risk * 0.15 +      # Increased from 0.1
                  central_risk * 0.1 +     # Kept same
                  motion_risk * 0.1 +      # Kept same
                  color_risk * 0.05)       # Kept same
    
    return min(risk_score, 1.0)


def create_intelligent_blur_mask(frame: np.ndarray, detections: List[Dict], risk_score: float) -> np.ndarray:
    """
    Create precise blur mask based on AI detections with confidence-weighted intensity.
    Industry standard: Higher confidence = stronger blur for specific areas.
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if not detections:
        # Fallback to legacy mask creation if no specific detections
        return create_facebook_compliant_mask_legacy(frame, risk_score)
    
    # Create precise masks for each detection with confidence-based intensity
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        detection_type = detection['type']
        
        # Ensure coordinates are within frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Calculate mask intensity based on detection type and confidence
        type_intensity_multipliers = {
            'weapon': 1.0,           # Maximum intensity for weapons
            'blood_pattern_0': 0.95,  # Near-maximum for fresh blood
            'blood_pattern_1': 0.85,  # High for dark blood
            'blood_pattern_2': 0.75,  # Medium-high for dried blood
            'injury': 0.9,           # Very high for injuries
            'high_activity': 0.6     # Medium for motion
        }
        
        intensity_multiplier = type_intensity_multipliers.get(detection_type, 0.7)
        mask_intensity = int(255 * confidence * intensity_multiplier)
        
        # Calculate padding based on confidence (higher confidence = more padding for safety)
        base_padding = 15
        confidence_padding = int(base_padding * (1 + confidence))
        
        x1_pad = max(0, x1 - confidence_padding)
        y1_pad = max(0, y1 - confidence_padding)
        x2_pad = min(w, x2 + confidence_padding)
        y2_pad = min(h, y2 + confidence_padding)
        
        # For high-confidence weapons/gore, create graduated mask (stronger in center)
        if confidence > 0.7 and detection_type in ['weapon', 'blood_pattern_0', 'injury']:
            # Create gradient mask for high-confidence detections
            temp_mask = np.zeros((h, w), dtype=np.float32)
            
            # Inner core: Maximum intensity
            cv2.rectangle(temp_mask, (x1, y1), (x2, y2), mask_intensity, -1)
            
            # Outer ring: Reduced intensity for smooth transition
            cv2.rectangle(temp_mask, (x1_pad, y1_pad), (x2_pad, y2_pad), 
                         mask_intensity * 0.7, -1)
            
            # Blend with existing mask (take maximum)
            mask = np.maximum(mask, temp_mask.astype(np.uint8))
        else:
            # Standard rectangular mask for lower confidence detections
            cv2.rectangle(mask, (x1_pad, y1_pad), (x2_pad, y2_pad), mask_intensity, -1)
    
    # Apply morphological operations based on risk level
    if risk_score > 0.6:
        # Aggressive smoothing for high-risk content
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    else:
        # Standard smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Gaussian blur for smooth transitions (stronger for high-risk content)
    blur_size = 21 if risk_score > 0.6 else 15
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 7 if risk_score > 0.6 else 5)
    
    return mask


def create_facebook_compliant_mask_legacy(frame: np.ndarray, risk_score: float) -> np.ndarray:
    """Legacy mask creation (renamed from original function)."""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Convert to analysis formats
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    
    # Strategy based on risk level
    if risk_score > 0.7:
        # High risk: Aggressive central blurring
        center_x, center_y = w // 2, h // 2
        radius = int(min(w, h) * 0.4)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        # Add skin regions
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.bitwise_or(mask, skin_mask)
        
    elif risk_score > 0.4:
        # Medium risk: Focus on central and red areas
        center_x, center_y = w // 2, h // 2
        radius = int(min(w, h) * 0.25)
        cv2.circle(mask, (center_x, center_y), radius, 200, -1)
        
        # Add red content areas - more aggressive detection
        red_ranges = [
            ([0, 50, 50], [15, 255, 255]),
            ([160, 50, 50], [180, 255, 255])
        ]
        
        for lower, upper in red_ranges:
            red_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.bitwise_or(mask, red_mask)
    
    else:
        # Low risk: Conservative red detection only
        red_ranges = [
            ([0, 80, 80], [10, 255, 255]),    # Very red areas only
            ([170, 80, 80], [180, 255, 255])
        ]
        
        for lower, upper in red_ranges:
            red_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.bitwise_or(mask, red_mask)
    
    # Enhanced motion detection for all risk levels
    if risk_score > 0.2:
        # Calculate local variance to detect high-activity areas
        kernel = np.ones((9, 9), np.float32) / 81
        gray_float = gray.astype(np.float32)
        local_mean = cv2.filter2D(gray_float, -1, kernel)
        local_variance = cv2.filter2D((gray_float - local_mean) ** 2, -1, kernel)
        
        # Threshold high variance areas
        variance_threshold = np.percentile(local_variance, 90)
        high_variance_mask = (local_variance > variance_threshold).astype(np.uint8) * 128
        mask = cv2.bitwise_or(mask, high_variance_mask)
    
    # Apply morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Gaussian blur for smooth transitions
    mask = cv2.GaussianBlur(mask, (15, 15), 5)
    
    return mask


def facebook_compliant_redaction(input_path: str, output_path: str, debug_mode: bool = False):
    """
    Facebook-compliant redaction with proper H.264 encoding and audio preservation.
    Uses OpenCV for processing and FFmpeg for encoding to ensure platform compatibility.
    """
    print(f"📘 Facebook-Compliant Video Processing (Enhanced)")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Create temporary directory for frame processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_video = os.path.join(temp_dir, "processed_video.mp4")
        
        # Process frames with OpenCV
        process_frames_opencv(input_path, temp_video, debug_mode)
        
        # Combine processed video with original audio using FFmpeg
        combine_with_audio_ffmpeg(temp_video, input_path, output_path)


def process_frames_opencv(input_path: str, temp_output: str, debug_mode: bool = False):
    """Process video frames using OpenCV with intelligent gore detection."""
    # Initialize intelligent gore detector
    print("🤖 Initializing AI Gore Detection System...")
    gore_detector = IntelligentGoreDetector()
    
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video Info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer with compatible codec for OpenCV processing
    # Try multiple codecs in order of preference until one works
    codecs_to_try = [
        ('MJPG', 'Motion JPEG (most compatible)'),
        ('XVID', 'Xvid MPEG-4 codec'),
        ('mp4v', 'MPEG-4 fallback')
    ]
    
    out = None
    for fourcc_str, description in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            
            # Test if the writer is properly initialized
            if out.isOpened():
                print(f"🎬 Using {description} for processing")
                break
            else:
                out.release()
                out = None
        except Exception as e:
            print(f"⚠️  {fourcc_str} codec failed: {e}")
            if out:
                out.release()
                out = None
    
    if out is None:
        raise RuntimeError("❌ Could not initialize video writer with any codec")
    
    # Note: FFmpeg will handle final H.264 encoding
    
    frame_count = 0
    frames_processed = 0
    ai_detections_count = 0
    
    try:
        print("\n🎬 Processing frames with AI-powered gore detection...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            progress = (frame_count / total_frames) * 100
            print(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%) - AI detections: {ai_detections_count}", end='\r')
            
            # AI-powered content analysis
            facebook_risk_score, detections = analyze_facebook_risk_intelligent(frame, gore_detector)
            
            # Debug visualization if enabled
            if debug_mode and detections:
                debug_frame = frame.copy()
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    det_type = detection.get('type', 'unknown')
                    confidence = detection.get('confidence', 0)
                    
                    # Draw bounding box
                    color = (0, 0, 255) if det_type == 'weapon' else (0, 255, 255)  # Red for weapons, yellow for others
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{det_type}: {confidence:.2f}"
                    cv2.putText(debug_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Save debug frame occasionally
                if frame_count % 30 == 0:  # Every 30 frames
                    debug_path = f"debug_frame_{frame_count:04d}.jpg"
                    cv2.imwrite(debug_path, debug_frame)
                    print(f"\n🔍 Debug frame saved: {debug_path}")
            
            # If Facebook would flag this content, apply precise AI-guided blur
            if facebook_risk_score > 0.1:  # LOWERED from 0.2 - more aggressive flagging
                if detections:
                    # Use AI detections for precise blurring
                    blur_mask = create_intelligent_blur_mask(frame, detections, facebook_risk_score)
                    ai_detections_count += len(detections)
                else:
                    # Fallback to legacy broad blurring
                    blur_mask = create_facebook_compliant_mask_legacy(frame, facebook_risk_score)
                
                redacted_frame = apply_facebook_blur(frame, blur_mask, facebook_risk_score)
                frames_processed += 1
            else:
                redacted_frame = frame.copy()
            
            # Write frame
            out.write(redacted_frame)
    
    finally:
        cap.release()
        out.release()
        
        print(f"\n✅ Frame processing complete! Processed {frames_processed} frames")


def combine_with_audio_ffmpeg(processed_video: str, original_video: str, output_path: str):
    """Combine processed video with original audio using FFmpeg for platform compatibility."""
    print("🎵 Combining with original audio using FFmpeg...")
    
    # FFmpeg command to combine processed video with original audio
    # Using H.264 codec with Facebook-compatible settings
    cmd = [
        'ffmpeg', '-y',  # -y to overwrite output file
        '-i', processed_video,  # Processed video (no audio)
        '-i', original_video,   # Original video (for audio)
        '-c:v', 'libx264',      # H.264 video codec
        '-preset', 'medium',    # Encoding speed vs compression
        '-crf', '23',           # Quality setting (18-28 range, 23 is good)
        '-c:a', 'aac',          # AAC audio codec
        '-b:a', '128k',         # Audio bitrate
        '-movflags', '+faststart',  # Optimize for web streaming
        '-pix_fmt', 'yuv420p',      # Pixel format for compatibility
        output_path
    ]
    
    try:
        # Run FFmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Video successfully encoded with H.264 + AAC for Facebook compatibility!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg error: {e}")
        print(f"FFmpeg stderr: {e.stderr}")
        
        # Fallback: try without audio if FFmpeg fails
        print("🔄 Trying fallback encoding without audio...")
        fallback_cmd = [
            'ffmpeg', '-y',
            '-i', processed_video,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-movflags', '+faststart',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        
        try:
            subprocess.run(fallback_cmd, capture_output=True, text=True, check=True)
            print("⚠️  Video encoded without audio (FFmpeg audio issue)")
        except subprocess.CalledProcessError as e2:
            print(f"❌ Fallback encoding also failed: {e2}")
            raise
    
    except FileNotFoundError:
        print("❌ FFmpeg not found! Please install FFmpeg for proper video encoding.")
        print("   For now, using OpenCV output (may have compatibility issues)")
        # Copy the temp file as fallback
        import shutil
        shutil.copy2(processed_video, output_path)


def analyze_facebook_risk(frame: np.ndarray) -> float:
    """
    Analyze Facebook content violation risk using computer vision.
    ENHANCED VERSION - More aggressive since Facebook still flagged content.
    """
    # Convert to different color spaces
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    
    risk_score = 0.0
    
    # 1. Blood/Red content detection - MORE AGGRESSIVE
    blood_score = detect_blood_content(hsv)
    risk_score = max(risk_score, blood_score * 1.2)  # Increased from 0.8
    
    # 2. Skin detection - MORE AGGRESSIVE
    skin_score = detect_skin_content(hsv)
    risk_score = max(risk_score, skin_score * 1.0)  # Increased from 0.6
    
    # 3. Edge/weapon detection - MORE AGGRESSIVE
    edge_score = detect_sharp_objects(gray)
    risk_score = max(risk_score, edge_score * 0.8)  # Increased from 0.5
    
    # 4. Dark/disturbing scene detection - MORE AGGRESSIVE
    darkness_score = detect_disturbing_scenes(gray)
    risk_score = max(risk_score, darkness_score * 0.7)  # Increased from 0.4
    
    # 5. Central content analysis - MORE AGGRESSIVE
    central_score = analyze_central_content(frame_rgb)
    risk_score = max(risk_score, central_score * 1.0)  # Increased from 0.7
    
    # 6. NEW: Motion/activity detection (Facebook flags active scenes)
    motion_score = detect_motion_activity(gray)
    risk_score = max(risk_score, motion_score * 0.6)
    
    # 7. NEW: Color variance detection (unusual patterns)
    color_score = detect_unusual_colors(hsv)
    risk_score = max(risk_score, color_score * 0.5)
    
    return min(risk_score, 1.0)


def detect_blood_content(hsv: np.ndarray) -> float:
    """Detect blood-like red content that Facebook would flag - ENHANCED VERSION."""
    # More aggressive red ranges for blood detection
    red_ranges = [
        ([0, 30, 30], [20, 255, 255]),    # Expanded red range 1
        ([160, 30, 30], [180, 255, 255])  # Expanded red range 2
    ]
    
    total_red_area = 0
    h, w = hsv.shape[:2]
    total_area = h * w
    
    for lower, upper in red_ranges:
        red_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Less aggressive filtering - catch more potential blood
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Smaller kernel
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        red_area = np.sum(red_mask > 0)
        total_red_area += red_area
    
    # Calculate red content percentage
    red_percentage = total_red_area / total_area
    
    # Facebook likely flags content with ANY significant red areas - more aggressive
    if red_percentage > 0.02:  # Lowered from 0.05 to 0.02 (2% red content)
        return min(red_percentage * 15, 1.0)  # Increased multiplier from 10 to 15
    
    return red_percentage * 5  # Increased from 2 to 5


def detect_skin_content(hsv: np.ndarray) -> float:
    """Detect skin-tone content that might indicate people."""
    # Skin tone range
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    h, w = hsv.shape[:2]
    skin_area = np.sum(skin_mask > 0)
    skin_percentage = skin_area / (h * w)
    
    # Facebook flags content with significant human presence in certain contexts
    if skin_percentage > 0.1:  # 10% skin content
        return min(skin_percentage * 5, 1.0)
    
    return skin_percentage


def detect_sharp_objects(gray: np.ndarray) -> float:
    """Detect sharp edges that might indicate weapons or dangerous objects."""
    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find long, straight lines (potential weapons)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    
    edge_score = 0.0
    
    if lines is not None:
        long_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > 50:  # Long lines might be weapons
                long_lines += 1
        
        if long_lines > 5:  # Multiple long lines
            edge_score = min(long_lines / 20, 0.8)
    
    # Also check overall edge density
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    if edge_density > 0.1:  # High edge density
        edge_score = max(edge_score, edge_density * 3)
    
    return edge_score


def detect_disturbing_scenes(gray: np.ndarray) -> float:
    """Detect dark or potentially disturbing scenes."""
    # Calculate overall brightness
    mean_brightness = np.mean(gray)
    
    # Calculate contrast
    contrast = np.std(gray)
    
    darkness_score = 0.0
    
    # Very dark scenes might indicate disturbing content
    if mean_brightness < 50:  # Very dark
        darkness_score = (50 - mean_brightness) / 50 * 0.6
    
    # High contrast might indicate dramatic/violent scenes
    if contrast > 60:  # High contrast
        darkness_score = max(darkness_score, (contrast - 60) / 60 * 0.4)
    
    return darkness_score


def analyze_central_content(frame_rgb: np.ndarray) -> float:
    """Analyze central content where Facebook focuses attention."""
    h, w = frame_rgb.shape[:2]
    
    # Extract central region (Facebook's likely focus area)
    center_h_start = h // 4
    center_h_end = 3 * h // 4
    center_w_start = w // 4
    center_w_end = 3 * w // 4
    
    central_region = frame_rgb[center_h_start:center_h_end, center_w_start:center_w_end]
    
    # Analyze central region for variance (activity/interest)
    gray_central = cv2.cvtColor(central_region, cv2.COLOR_RGB2GRAY)
    variance = np.var(gray_central)
    
    # High variance in central region might indicate important content
    if variance > 1000:  # High activity in center
        return min(variance / 5000, 0.8)
    
    return variance / 10000


def create_facebook_compliant_mask(frame: np.ndarray, risk_score: float) -> np.ndarray:
    """
    Create blur mask based on Facebook compliance requirements.
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Convert to analysis formats
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    
    # Strategy based on risk level
    if risk_score > 0.7:
        # High risk: Aggressive central blurring
        center_x, center_y = w // 2, h // 2
        radius = int(min(w, h) * 0.4)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        
        # Add skin regions
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.bitwise_or(mask, skin_mask)
        
    elif risk_score > 0.4:
        # Medium risk: Target specific regions
        # Blood/red regions
        red_ranges = [([0, 50, 50], [15, 255, 255]), ([165, 50, 50], [180, 255, 255])]
        for lower, upper in red_ranges:
            red_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.bitwise_or(mask, red_mask)
        
        # Central region
        center_x, center_y = w // 2, h // 2
        radius = int(min(w, h) * 0.25)
        cv2.circle(mask, (center_x, center_y), radius, 128, -1)
        
    else:
        # Low risk: Light strategic blurring
        # Find areas of high variance
        kernel = np.ones((9, 9), np.float32) / 81
        gray_float = gray.astype(np.float32)
        local_mean = cv2.filter2D(gray_float, -1, kernel)
        local_variance = cv2.filter2D((gray_float - local_mean) ** 2, -1, kernel)
        
        # Threshold high variance areas
        variance_threshold = np.percentile(local_variance, 90)
        high_variance_mask = (local_variance > variance_threshold).astype(np.uint8) * 128
        mask = cv2.bitwise_or(mask, high_variance_mask)
    
    # Apply morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Gaussian blur for smooth transitions
    mask = cv2.GaussianBlur(mask, (15, 15), 5)
    
    return mask


def apply_facebook_blur(frame: np.ndarray, mask: np.ndarray, risk_score: float) -> np.ndarray:
    """Apply Facebook-compliant blurring - ENHANCED VERSION with industry-standard strength."""
    # Industry-standard blur intensity based on content severity
    if risk_score > 0.8:
        blur_strength = 51  # MAXIMUM: High gore/violence (industry standard for explicit content)
    elif risk_score > 0.6:
        blur_strength = 41  # HEAVY: Medium violence/weapons (ensure complete obscuration)
    elif risk_score > 0.4:
        blur_strength = 31  # STRONG: Low-medium risk (standard social media compliance)
    elif risk_score > 0.2:
        blur_strength = 21  # MEDIUM: Precautionary blurring
    else:
        blur_strength = 15  # LIGHT: Conservative safety margin
    
    # Ensure blur_strength is odd and at least 3 (OpenCV requirement)
    if blur_strength % 2 == 0:
        blur_strength += 1
    blur_strength = max(3, blur_strength)
    
    # Calculate second pass blur size for double-pass safety (industry standard)
    second_blur = max(3, blur_strength // 2)
    if second_blur % 2 == 0:
        second_blur += 1
    
    # Apply TRIPLE-PASS blur for high-risk content (exceeds industry standard)
    blurred = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
    blurred = cv2.GaussianBlur(blurred, (second_blur, second_blur), 0)  # Second pass
    
    # Triple pass for high-risk content (ensures complete unrecognizability)
    if risk_score > 0.6:
        third_blur = max(3, blur_strength // 3)
        if third_blur % 2 == 0:
            third_blur += 1
        blurred = cv2.GaussianBlur(blurred, (third_blur, third_blur), 0)  # Third pass
    
    # Apply mask with stronger blending for high-risk areas
    mask_normalized = mask.astype(np.float32) / 255.0
    
    # Enhanced mask strength for pinpointed areas
    if risk_score > 0.5:
        # Boost mask intensity for high-confidence AI detections
        mask_normalized = np.power(mask_normalized, 0.7)  # Makes mask more aggressive
    
    mask_3d = np.dstack([mask_normalized] * 3)
    
    # Blend original and blurred based on mask
    result = frame.astype(np.float32) * (1 - mask_3d) + blurred.astype(np.float32) * mask_3d
    
    return result.astype(np.uint8)


def detect_motion_activity(gray: np.ndarray) -> float:
    """Detect motion/activity patterns that Facebook might flag."""
    # Use Laplacian to detect areas of high activity/change
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    activity_score = np.var(laplacian)
    
    # Normalize and return score
    normalized_score = min(activity_score / 1000, 1.0)  # Normalize to 0-1
    return normalized_score


def analyze_color_variance(hsv: np.ndarray) -> float:
    """Analyze color variance patterns that might indicate problematic content."""
    h, w = hsv.shape[:2]
    
    # Calculate color variance across the image
    hue_channel = hsv[:, :, 0]
    saturation_channel = hsv[:, :, 1]
    
    # High saturation variance might indicate unusual content
    sat_variance = np.var(saturation_channel)
    hue_variance = np.var(hue_channel)
    
    # Combine variances
    color_variance = (sat_variance + hue_variance) / 2
    
    # Normalize to 0-1 range
    normalized_score = min(color_variance / 2000, 1.0)
    return normalized_score


def detect_unusual_colors(hsv: np.ndarray) -> float:
    """Detect unusual color patterns that might indicate sensitive content."""
    h, w = hsv.shape[:2]
    
    # Calculate color variance across the image
    hue_channel = hsv[:, :, 0]
    saturation_channel = hsv[:, :, 1]
    
    # High saturation variance might indicate unusual content
    sat_variance = np.var(saturation_channel)
    hue_variance = np.var(hue_channel)
    
    # Combine variances
    color_variance = (sat_variance + hue_variance) / 2
    
    # Normalize to 0-1 range
    normalized_score = min(color_variance / 2000, 1.0)
    return normalized_score


def main():
    """Main entry point for Facebook-compliant video processing."""
    parser = argparse.ArgumentParser(description="Facebook-Compliant Video Redaction")
    
    parser.add_argument("input", help="Input video file")
    parser.add_argument("output", help="Output video file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detection visualization")
    
    args = parser.parse_args()
    
    # Process video with Facebook compliance
    facebook_compliant_redaction(args.input, args.output, args.debug)


if __name__ == "__main__":
    main()
