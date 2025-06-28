"""
Enhanced Configuration Management for Production Body Measurement System
======================================================================

Comprehensive configuration with multi-view settings, production parameters,
and advanced model configurations.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum

class ModelType(Enum):
    """Available model types"""
    YOLOV8 = "yolov8"
    YOLOV9 = "yolov9"
    YOLOV10 = "yolov10"
    MEDIAPIPE = "mediapipe"
    DETECTRON2 = "detectron2"
    MMPOSE = "mmpose"
    SAM = "segment_anything"
    MIDAS = "midas_depth"

class PoseModel(Enum):
    """Available pose estimation models"""
    MEDIAPIPE_POSE = "mediapipe_pose"
    YOLO_POSE = "yolo_pose"
    MMPOSE_HRNET = "mmpose_hrnet"
    MMPOSE_VITPOSE = "mmpose_vitpose"
    OPENPOSE = "openpose"
    ALPHAPOSE = "alphapose"

class ViewType(Enum):
    """Supported view types"""
    FRONT = "front"
    SIDE = "side"
    BACK = "back"
    THREE_QUARTER = "three_quarter"

class MeasurementPrecision(Enum):
    """Measurement precision levels"""
    STANDARD = "standard"    # ±1cm
    HIGH = "high"           # ±0.5cm
    ULTRA = "ultra"         # ±0.2cm
    RESEARCH = "research"   # ±0.1cm

@dataclass
class ModelConfig:
    """Enhanced model configuration"""
    # Primary models for ensemble
    primary_pose_model: str = "yolov8x-pose.pt"
    secondary_pose_model: str = "yolov8l-pose.pt"
    tertiary_pose_model: str = "yolov8m-pose.pt"
    
    # Backwards compatibility
    primary_detector: str = "yolov8x-pose.pt"
    secondary_detector: str = "yolov8l-pose.pt"
    
    # Specialized models
    segmentation_model: str = "yolov8x-seg.pt"
    depth_estimation_model: str = "midas_dpt_large"
    face_detection_model: str = "yolov8n-face.pt"
    
    # MMPose configuration
    mmpose_config: str = "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py"
    mmpose_checkpoint: str = "hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
    
    # Model paths and caching
    model_cache_dir: str = "models"
    auto_download: bool = True
    download_models: bool = True  # Backwards compatibility
    model_update_check: bool = False
    
    # Pose settings (backwards compatibility)
    pose_model: str = "mediapipe_pose"
    pose_confidence_threshold: float = 0.5
    pose_tracking: bool = True
    
    # Performance settings
    use_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    mixed_precision: bool = True
    half_precision: bool = True  # Backwards compatibility
    batch_size: int = 1
    num_workers: int = 4
    
    # Model ensemble weights
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "yolo_x": 0.4,
        "yolo_l": 0.2,
        "mediapipe": 0.3,
        "mmpose": 0.1
    })
    
    # Advanced features
    temporal_smoothing: bool = True
    outlier_detection: bool = True
    multi_scale_inference: bool = True
    test_time_augmentation: bool = False
    
    # Advanced models
    use_sam: bool = False  # Segment Anything Model
    use_dino: bool = False  # DINO for fine-grained detection
    use_depth_estimation: bool = False  # Depth estimation models

@dataclass
class MultiViewConfig:
    """Multi-view analysis configuration"""
    # Required views for high accuracy
    required_views: List[str] = field(default_factory=lambda: ["front", "side"])
    optional_views: List[str] = field(default_factory=lambda: ["back"])
    
    # View validation
    auto_view_classification: bool = True
    view_classification_confidence_threshold: float = 0.8
    cross_view_validation: bool = True
    
    # 3D reconstruction
    enable_3d_reconstruction: bool = True
    stereo_calibration_file: Optional[str] = None
    camera_intrinsics: Optional[Dict[str, Any]] = None
    
    # Multi-view fusion
    fusion_method: str = "weighted_average"  # "weighted_average", "kalman_filter", "particle_filter"
    temporal_consistency: bool = True
    geometric_constraints: bool = True
    
    # Quality requirements
    minimum_views_for_measurement: int = 2
    minimum_pose_quality_per_view: str = "fair"
    cross_view_consistency_threshold: float = 0.7

@dataclass
class MeasurementConfig:
    """Enhanced measurement configuration"""
    # Target precision
    target_precision: str = "high"  # standard, high, ultra, research
    
    # Reference measurements
    default_height_cm: float = 170.0
    height_range_cm: tuple = (140.0, 220.0)
    
    # Body measurements to calculate
    body_parts: List[str] = field(default_factory=lambda: [
        "height", "shoulder_width", "chest_circumference", "waist_circumference",
        "hip_circumference", "arm_length", "leg_length", "inseam", "torso_length",
        "neck_circumference", "bicep_circumference", "thigh_circumference",
        "wrist_circumference", "ankle_circumference", "arm_span", "sitting_height"
    ])
    
    # Advanced measurements
    advanced_measurements: List[str] = field(default_factory=lambda: [
        "shoulder_slope", "back_width", "chest_depth", "waist_depth",
        "hip_depth", "crotch_height", "knee_height", "ankle_height"
    ])
    
    # Measurement precision settings
    measurement_precision_digits: int = 1
    measurement_precision: int = 2  # For backwards compatibility
    uncertainty_calculation: bool = True
    confidence_weighting: bool = True
    
    # Backwards compatibility attributes
    confidence_threshold: float = 0.7
    
    # Calibration settings
    calibration_method: str = "multi_point"  # "head_to_toe", "multi_point", "object_reference"
    auto_calibrate: bool = True
    calibration_confidence_threshold: float = 0.8
    
    # Validation and filtering
    anthropometric_validation: bool = True
    outlier_detection_method: str = "iqr"  # "iqr", "zscore", "isolation_forest"
    measurement_smoothing: bool = True
    apply_smoothing: bool = True  # Backwards compatibility
    outlier_detection: bool = True  # Backwards compatibility
    
    # Population data for validation
    population_database: str = "anthropometric_data/comprehensive_db.json"
    age_range: tuple = (18, 65)
    gender_specific_validation: bool = True

@dataclass
class ProcessingConfig:
    """Image and data processing configuration"""
    # Image preprocessing
    target_resolution: tuple = (1280, 1280)
    maintain_aspect_ratio: bool = True
    image_enhancement: bool = True
    
    # Enhancement parameters
    contrast_enhancement: float = 1.2
    brightness_adjustment: float = 0.0
    gamma_correction: float = 1.0
    noise_reduction: bool = True
    
    # Multi-scale processing
    enable_multi_scale: bool = True
    scale_factors: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2])
    
    # Post-processing
    nms_threshold: float = 0.5
    confidence_threshold: float = 0.7
    keypoint_confidence_threshold: float = 0.5
    
    # Quality control
    blur_detection: bool = True
    blur_threshold: float = 100.0
    exposure_check: bool = True
    
    # Performance optimization
    parallel_processing: bool = True
    memory_optimization: bool = True
    cpu_optimization: bool = True

@dataclass
class ProductionConfig:
    """Production environment configuration"""
    # Environment
    environment: str = "production"  # "development", "staging", "production"
    debug_mode: bool = False
    verbose_logging: bool = False
    
    # Performance monitoring
    enable_metrics: bool = True
    metrics_interval_seconds: int = 60
    performance_logging: bool = True
    
    # Error handling
    retry_attempts: int = 3
    timeout_seconds: int = 30
    graceful_degradation: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_size_mb: int = 512
    cache_ttl_seconds: int = 3600
    
    # Security
    input_validation: bool = True
    sanitize_file_paths: bool = True
    max_file_size_mb: int = 50
    allowed_file_extensions: List[str] = field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
    ])
    
    # Rate limiting
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    max_concurrent_requests: int = 10
    
    # Database configuration
    database_url: Optional[str] = None
    connection_pool_size: int = 10
    query_timeout_seconds: int = 30

@dataclass
class UIConfig:
    """User interface configuration"""
    # Window settings
    window_title: str = "AI Body Measurement System - Professional Edition"
    window_size: tuple = (1600, 1000)
    theme: str = "professional"  # "light", "dark", "professional"
    
    # Multi-view interface
    multi_view_layout: str = "grid"  # "grid", "tabbed", "carousel"
    show_view_labels: bool = True
    auto_arrange_views: bool = True
    
    # Visualization options
    show_skeleton: bool = True
    show_keypoints: bool = True
    show_confidence_scores: bool = True
    show_3d_overlay: bool = False
    show_measurements_overlay: bool = True
    
    # Colors and styling
    skeleton_color: tuple = (0, 255, 0)
    keypoint_colors: Dict[str, tuple] = field(default_factory=lambda: {
        "high_confidence": (0, 255, 0),      # Green
        "medium_confidence": (0, 255, 255),  # Yellow
        "low_confidence": (0, 0, 255)        # Red
    })
    
    # Professional features
    measurement_precision_display: int = 2
    show_uncertainty_ranges: bool = True
    real_time_feedback: bool = True
    professional_reporting: bool = True

@dataclass
class ExportConfig:
    """Export and reporting configuration"""
    # Output formats
    supported_formats: List[str] = field(default_factory=lambda: [
        "json", "csv", "excel", "pdf", "xml"
    ])
    default_format: str = "json"
    
    # File organization
    output_directory: str = "outputs"
    create_session_folders: bool = True
    timestamp_files: bool = True
    
    # Report generation
    generate_detailed_report: bool = True
    include_quality_metrics: bool = True
    include_confidence_intervals: bool = True
    include_comparison_charts: bool = True
    
    # Report templates
    report_template: str = "professional"  # "basic", "detailed", "professional", "research"
    company_branding: bool = False
    custom_logo_path: Optional[str] = None
    
    # Data retention
    retain_processed_images: bool = True
    retain_intermediate_results: bool = False
    data_retention_days: int = 90
    
    # Privacy and compliance
    anonymize_exports: bool = True
    comply_with_gdpr: bool = True
    audit_trail: bool = True

class EnhancedConfig:
    """Enhanced configuration management for production system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration"""
        self.config_path = config_path
        
        # Initialize all configuration sections
        self.model = ModelConfig()
        self.multi_view = MultiViewConfig()
        self.measurement = MeasurementConfig()
        self.processing = ProcessingConfig()
        self.production = ProductionConfig()
        self.ui = UIConfig()
        self.export = ExportConfig()
        
        # Load configuration if path provided
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
        # Validate configuration
        self._validate_config()
        
        # Create necessary directories
        self._create_directories()
    
    def _validate_config(self):
        """Validate configuration settings"""
        
        # Validate precision settings
        valid_precisions = [p.value for p in MeasurementPrecision]
        if self.measurement.target_precision not in valid_precisions:
            raise ValueError(f"Invalid precision: {self.measurement.target_precision}")
        
        # Validate required views
        valid_views = [v.value for v in ViewType]
        for view in self.multi_view.required_views:
            if view not in valid_views:
                raise ValueError(f"Invalid required view: {view}")
        
        # Validate file paths
        if self.export.custom_logo_path and not os.path.exists(self.export.custom_logo_path):
            raise FileNotFoundError(f"Custom logo not found: {self.export.custom_logo_path}")
        
        # Validate numeric ranges
        if self.measurement.height_range_cm[0] >= self.measurement.height_range_cm[1]:
            raise ValueError("Invalid height range")
        
        # Production environment validations
        if self.production.environment == "production":
            if self.production.debug_mode:
                raise ValueError("Debug mode cannot be enabled in production")
            if not self.production.input_validation:
                raise ValueError("Input validation must be enabled in production")
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.model.model_cache_dir,
            self.export.output_directory,
            "logs",
            "temp",
            "cache",
            "reports"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_config(self, config_path: str):
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        # Update configuration sections
        for section_name, section_data in config_data.items():
            if hasattr(self, section_name):
                self._update_config_section(section_name, section_data)
    
    def _update_config_section(self, section_name: str, config_data: Dict[str, Any]):
        """Update a configuration section"""
        section = getattr(self, section_name)
        
        for key, value in config_data.items():
            if hasattr(section, key):
                # Handle nested dictionaries and lists properly
                current_value = getattr(section, key)
                if isinstance(current_value, dict) and isinstance(value, dict):
                    current_value.update(value)
                else:
                    setattr(section, key, value)
    
    def save_config(self, config_path: str):
        """Save configuration to file"""
        config_data = {
            'model': asdict(self.model),
            'multi_view': asdict(self.multi_view),
            'measurement': asdict(self.measurement),
            'processing': asdict(self.processing),
            'production': asdict(self.production),
            'ui': asdict(self.ui),
            'export': asdict(self.export)
        }
        
        config_path = Path(config_path)
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    def get_model_path(self, model_name: str) -> str:
        """Get full path to a model file"""
        return os.path.join(self.model.model_cache_dir, model_name)
    
    def get_output_path(self, filename: str, create_session_folder: bool = None) -> str:
        """Get full path to an output file"""
        if create_session_folder is None:
            create_session_folder = self.export.create_session_folders
        
        base_path = Path(self.export.output_directory)
        
        if create_session_folder:
            from datetime import datetime
            session_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = base_path / session_folder
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = base_path
        
        return str(output_path / filename)
    
    def get_precision_settings(self) -> Dict[str, float]:
        """Get precision settings for current target precision"""
        
        precision_settings = {
            "standard": {
                "target_uncertainty": 1.0,  # ±1cm
                "min_confidence": 0.7,
                "min_keypoints": 8,
                "min_views": 1
            },
            "high": {
                "target_uncertainty": 0.5,  # ±0.5cm
                "min_confidence": 0.8,
                "min_keypoints": 12,
                "min_views": 2
            },
            "ultra": {
                "target_uncertainty": 0.2,  # ±0.2cm
                "min_confidence": 0.9,
                "min_keypoints": 15,
                "min_views": 3
            },
            "research": {
                "target_uncertainty": 0.1,  # ±0.1cm
                "min_confidence": 0.95,
                "min_keypoints": 17,
                "min_views": 3
            }
        }
        
        return precision_settings.get(self.measurement.target_precision, precision_settings["standard"])
    
    def adapt_for_environment(self, environment: str):
        """Adapt configuration for specific environment"""
        
        if environment == "development":
            self.production.debug_mode = True
            self.production.verbose_logging = True
            self.production.retry_attempts = 1
            self.model.auto_download = True
            self.processing.parallel_processing = False
            
        elif environment == "staging":
            self.production.debug_mode = False
            self.production.verbose_logging = True
            self.production.enable_metrics = True
            self.production.retry_attempts = 2
            
        elif environment == "production":
            self.production.debug_mode = False
            self.production.verbose_logging = False
            self.production.enable_metrics = True
            self.production.retry_attempts = 3
            self.production.input_validation = True
            self.production.enable_rate_limiting = True
            
        self.production.environment = environment
    
    def optimize_for_hardware(self, gpu_available: bool = None, gpu_memory_gb: float = None,
                            cpu_cores: int = None, ram_gb: float = None):
        """Optimize configuration based on available hardware"""
        
        if gpu_available is None:
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                if gpu_available and gpu_memory_gb is None:
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except ImportError:
                gpu_available = False
        
        # GPU optimization
        self.model.use_gpu = gpu_available
        
        if gpu_available and gpu_memory_gb:
            if gpu_memory_gb >= 8:
                self.model.mixed_precision = True
                self.model.batch_size = 2
                self.processing.enable_multi_scale = True
            elif gpu_memory_gb >= 4:
                self.model.mixed_precision = True
                self.model.batch_size = 1
                self.processing.enable_multi_scale = False
            else:
                self.model.mixed_precision = False
                self.model.batch_size = 1
                self.processing.enable_multi_scale = False
            
            # Adjust memory fraction
            self.model.gpu_memory_fraction = min(0.8, (gpu_memory_gb - 1) / gpu_memory_gb)
        
        # CPU optimization
        if cpu_cores:
            self.model.num_workers = min(cpu_cores, 8)
            self.processing.parallel_processing = cpu_cores > 4
        
        # RAM optimization
        if ram_gb:
            if ram_gb >= 16:
                self.production.cache_size_mb = 1024
                self.processing.memory_optimization = False
            elif ram_gb >= 8:
                self.production.cache_size_mb = 512
                self.processing.memory_optimization = True
            else:
                self.production.cache_size_mb = 256
                self.processing.memory_optimization = True
                self.processing.parallel_processing = False
    
    def get_measurement_config_for_view(self, view_type: str) -> Dict[str, Any]:
        """Get measurement configuration optimized for specific view"""
        
        base_config = {
            "body_parts": self.measurement.body_parts.copy(),
            "precision": self.measurement.target_precision,
            "confidence_threshold": self.processing.confidence_threshold
        }
        
        # View-specific optimizations
        if view_type == "front":
            base_config["primary_measurements"] = [
                "shoulder_width", "chest_circumference", "waist_circumference",
                "hip_circumference", "arm_span"
            ]
            base_config["confidence_boost"] = 1.1
            
        elif view_type == "side":
            base_config["primary_measurements"] = [
                "height", "torso_length", "arm_length", "leg_length",
                "chest_depth", "waist_depth"
            ]
            base_config["confidence_boost"] = 1.0
            
        elif view_type == "back":
            base_config["primary_measurements"] = [
                "shoulder_width", "back_width", "shoulder_slope"
            ]
            base_config["confidence_boost"] = 0.9
        
        return base_config
    
    def validate_multi_view_setup(self, available_views: List[str]) -> Dict[str, Any]:
        """Validate if available views meet requirements"""
        
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommended_precision": self.measurement.target_precision
        }
        
        # Check required views
        missing_required = set(self.multi_view.required_views) - set(available_views)
        if missing_required:
            validation_result["errors"].append(
                f"Missing required views: {list(missing_required)}"
            )
            validation_result["valid"] = False
        
        # Check minimum views for target precision
        precision_settings = self.get_precision_settings()
        min_views_needed = precision_settings["min_views"]
        
        if len(available_views) < min_views_needed:
            validation_result["warnings"].append(
                f"Only {len(available_views)} views available, "
                f"{min_views_needed} needed for {self.measurement.target_precision} precision"
            )
            
            # Recommend lower precision
            if len(available_views) >= 2:
                validation_result["recommended_precision"] = "high"
            elif len(available_views) >= 1:
                validation_result["recommended_precision"] = "standard"
            else:
                validation_result["valid"] = False
                validation_result["errors"].append("No views available")
        
        # Check view combination quality
        if "front" in available_views and "side" in available_views:
            validation_result["quality_score"] = 1.0
        elif "front" in available_views or "side" in available_views:
            validation_result["quality_score"] = 0.8
        else:
            validation_result["quality_score"] = 0.6
            validation_result["warnings"].append(
                "Suboptimal view combination - front and side views recommended"
            )
        
        return validation_result
    
    def create_deployment_config(self, deployment_type: str = "standalone") -> Dict[str, Any]:
        """Create deployment-specific configuration"""
        
        deployment_configs = {
            "standalone": {
                "enable_api": False,
                "enable_web_interface": True,
                "enable_batch_processing": True,
                "max_concurrent_sessions": 1,
                "resource_limits": {
                    "max_memory_mb": 4096,
                    "max_processing_time_seconds": 300
                }
            },
            "server": {
                "enable_api": True,
                "enable_web_interface": True,
                "enable_batch_processing": True,
                "max_concurrent_sessions": 10,
                "resource_limits": {
                    "max_memory_mb": 8192,
                    "max_processing_time_seconds": 60
                }
            },
            "cloud": {
                "enable_api": True,
                "enable_web_interface": False,
                "enable_batch_processing": True,
                "max_concurrent_sessions": 100,
                "resource_limits": {
                    "max_memory_mb": 16384,
                    "max_processing_time_seconds": 30
                },
                "auto_scaling": True,
                "load_balancing": True
            },
            "edge": {
                "enable_api": True,
                "enable_web_interface": False,
                "enable_batch_processing": False,
                "max_concurrent_sessions": 1,
                "resource_limits": {
                    "max_memory_mb": 2048,
                    "max_processing_time_seconds": 120
                },
                "model_optimization": "quantized"
            }
        }
        
        base_config = deployment_configs.get(deployment_type, deployment_configs["standalone"])
        
        # Apply deployment-specific optimizations
        if deployment_type == "edge":
            self.model.mixed_precision = True
            self.model.ensemble_weights = {"yolo_m": 0.7, "mediapipe": 0.3}
            self.processing.enable_multi_scale = False
            
        elif deployment_type == "cloud":
            self.production.enable_metrics = True
            self.production.enable_rate_limiting = True
            self.production.max_concurrent_requests = 50
            
        return base_config
    
    def export_config_summary(self) -> str:
        """Export a human-readable configuration summary"""
        
        summary = f"""
AI Body Measurement System Configuration Summary
==============================================

Environment: {self.production.environment}
Target Precision: {self.measurement.target_precision}
Multi-view Enabled: {self.multi_view.enable_3d_reconstruction}

Model Configuration:
- Primary Pose Model: {self.model.primary_pose_model}
- GPU Enabled: {self.model.use_gpu}
- Mixed Precision: {self.model.mixed_precision}
- Ensemble Models: {len([k for k, v in self.model.ensemble_weights.items() if v > 0])}

Multi-view Settings:
- Required Views: {', '.join(self.multi_view.required_views)}
- 3D Reconstruction: {self.multi_view.enable_3d_reconstruction}
- Cross-view Validation: {self.multi_view.cross_view_validation}

Measurement Settings:
- Body Parts: {len(self.measurement.body_parts)} standard + {len(self.measurement.advanced_measurements)} advanced
- Anthropometric Validation: {self.measurement.anthropometric_validation}
- Precision Digits: {self.measurement.measurement_precision_digits}

Processing Settings:
- Target Resolution: {self.processing.target_resolution}
- Multi-scale Processing: {self.processing.enable_multi_scale}
- Parallel Processing: {self.processing.parallel_processing}

Production Settings:
- Debug Mode: {self.production.debug_mode}
- Metrics Enabled: {self.production.enable_metrics}
- Rate Limiting: {self.production.enable_rate_limiting}
- Caching: {self.production.enable_caching} ({self.production.cache_size_mb}MB)
"""
        
        precision_settings = self.get_precision_settings()
        summary += f"""
Current Precision Requirements:
- Target Uncertainty: ±{precision_settings['target_uncertainty']}cm
- Minimum Confidence: {precision_settings['min_confidence']}
- Minimum Keypoints: {precision_settings['min_keypoints']}
- Minimum Views: {precision_settings['min_views']}
"""
        
        return summary.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': asdict(self.model),
            'multi_view': asdict(self.multi_view),
            'measurement': asdict(self.measurement),
            'processing': asdict(self.processing),
            'production': asdict(self.production),
            'ui': asdict(self.ui),
            'export': asdict(self.export)
        }
    
    def clone(self) -> 'EnhancedConfig':
        """Create a deep copy of the configuration"""
        new_config = EnhancedConfig()
        
        # Copy all dataclass fields
        for section_name in ['model', 'multi_view', 'measurement', 'processing', 'production', 'ui', 'export']:
            source_section = getattr(self, section_name)
            target_section = getattr(new_config, section_name)
            
            for field_name, field_value in asdict(source_section).items():
                setattr(target_section, field_name, field_value)
        
        return new_config
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"EnhancedConfig(precision={self.measurement.target_precision}, " \
               f"environment={self.production.environment}, " \
               f"multi_view={self.multi_view.enable_3d_reconstruction})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__()

# Global configuration instance
_global_config = None

def get_config() -> EnhancedConfig:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = EnhancedConfig()
    return _global_config

def set_config(config: EnhancedConfig):
    """Set global configuration instance"""
    global _global_config
    _global_config = config

def load_config_from_file(config_path: str) -> EnhancedConfig:
    """Load configuration from file and set as global"""
    config = EnhancedConfig(config_path)
    set_config(config)
    return config

# Configuration presets for common use cases
def create_research_config() -> EnhancedConfig:
    """Create configuration optimized for research use"""
    config = EnhancedConfig()
    config.measurement.target_precision = "research"
    config.multi_view.required_views = ["front", "side", "back"]
    config.model.test_time_augmentation = True
    config.processing.enable_multi_scale = True
    config.production.enable_metrics = True
    return config

def create_production_config() -> EnhancedConfig:
    """Create configuration optimized for production deployment"""
    config = EnhancedConfig()
    config.measurement.target_precision = "high"
    config.production.environment = "production"
    config.production.debug_mode = False
    config.production.enable_rate_limiting = True
    config.production.input_validation = True
    return config

def create_mobile_config() -> EnhancedConfig:
    """Create configuration optimized for mobile/edge deployment"""
    config = EnhancedConfig()
    config.measurement.target_precision = "standard"
    config.model.ensemble_weights = {"yolo_m": 0.8, "mediapipe": 0.2}
    config.processing.target_resolution = (640, 640)
    config.processing.enable_multi_scale = False
    config.production.cache_size_mb = 128
    return config