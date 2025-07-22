#!/usr/bin/env python3
"""
Example usage of the new item system for the Reassembly application.
This file demonstrates how to create and manage different types of items.
"""

import os
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from left_panel.item_tree.items import (
    BaseItem, BaseModelItem, SegmentationResultItem, 
    ClassificationResultItem, PairwiseResultItem, AssemblyResultItem, PreprocessedItem
)


def example_preprocessed_item():
    """Example of creating and using a PreprocessedItem."""
    print("=== PreprocessedItem Example ===")
    
    # Create a preprocessed item
    item = PreprocessedItem(
        label="Preprocessed Model",
        preprocessed_mesh=None,  # Would be an Open3D mesh object
        original_mesh_path="/path/to/original.obj",
        preprocessing_parameters={
            "noise_reduction": True,
            "smoothing_factor": 0.1,
            "decimation_ratio": 0.5,
            "normal_estimation": True
        },
        preprocessing_steps=["noise_reduction", "smoothing", "decimation", "normal_estimation"],
        quality_metrics={
            "surface_smoothness": 0.85,
            "mesh_consistency": 0.92,
            "vertex_density": 0.78
        },
        is_visible=True
    )
    
    print(f"Item ID: {item.id}")
    print(f"Label: {item.label}")
    print(f"Original Mesh Path: {item.original_mesh_path}")
    print(f"Processing Time: {item.processing_time}s")
    print(f"Quality Score: {item.mesh_quality_score}")
    print(f"Preprocessing Steps: {item.preprocessing_steps}")
    print(f"Step Count: {item.get_step_count()}")
    print(f"Quality Metrics: {item.quality_metrics}")
    
    # Add more preprocessing steps
    item.add_preprocessing_step("boundary_detection")
    item.add_preprocessing_step("hole_filling")
    print(f"Updated Steps: {item.preprocessing_steps}")
    print(f"Updated Step Count: {item.get_step_count()}")
    
    # Add quality metrics
    item.add_quality_metric("boundary_quality", 0.88)
    item.add_quality_metric("hole_fill_quality", 0.91)
    print(f"Updated Quality Metrics: {item.quality_metrics}")
    
    # Set processing time and quality score
    item.processing_time = 2.5
    item.mesh_quality_score = 0.87
    print(f"Processing Time: {item.processing_time}s")
    print(f"Quality Score: {item.mesh_quality_score}")
    
    # Check if specific steps were applied
    print(f"Has noise reduction: {item.has_step('noise_reduction')}")
    print(f"Has texture_mapping: {item.has_step('texture_mapping')}")
    
    # Get processing summary
    summary = item.get_processing_summary()
    print(f"Processing Summary: {summary}")
    
    # Convert to dictionary
    item_dict = item.to_dict()
    print(f"Dictionary representation: {item_dict}")
    print()


def example_base_model_item():
    """Example of creating and using a BaseModelItem."""
    print("=== BaseModelItem Example ===")
    
    # Create a base model item
    item = BaseModelItem(
        label="example_mesh.obj",
        mesh_path="/path/to/example_mesh.obj",
        mesh=None,  # Would be an Open3D mesh object
        is_visible=True
    )
    
    print(f"Item ID: {item.id}")
    print(f"Label: {item.label}")
    print(f"Visible: {item.is_visible}")
    print(f"Mesh Path: {item.mesh_path}")
    print(f"File Name: {item.get_file_name()}")
    print(f"File Extension: {item.get_file_extension()}")
    print(f"Item Type: {item.get_item_type()}")
    
    # Change properties
    item.label = "Updated Label"
    item.is_visible = False
    print(f"Updated Label: {item.label}")
    print(f"Updated Visible: {item.is_visible}")
    print(f"Is Hidden: {item.is_hidden}")
    
    # Convert to dictionary
    item_dict = item.to_dict()
    print(f"Dictionary representation: {item_dict}")
    print()


def example_segmentation_item():
    """Example of creating and using a SegmentationResultItem."""
    print("=== SegmentationResultItem Example ===")
    
    # Create a segmentation result item
    item = SegmentationResultItem(
        label="Segmented Model",
        segmented_mesh=None,  # Would be an Open3D mesh object
        original_mesh_path="/path/to/original.obj",
        segmentation_parameters={
            "method": "region_growing",
            "threshold": 0.1,
            "min_segment_size": 100
        },
        is_visible=True
    )
    
    print(f"Item ID: {item.id}")
    print(f"Label: {item.label}")
    print(f"Original Mesh Path: {item.original_mesh_path}")
    print(f"Segment Count: {item.segment_count}")
    print(f"Segmentation Parameters: {item.segmentation_parameters}")
    
    # Add segment labels
    item.add_segment_label("Segment_1")
    item.add_segment_label("Segment_2")
    item.add_segment_label("Segment_3")
    print(f"Segment Labels: {item.segment_labels}")
    print(f"Updated Segment Count: {item.segment_count}")
    
    # Convert to dictionary
    item_dict = item.to_dict()
    print(f"Dictionary representation: {item_dict}")
    print()


def example_classification_item():
    """Example of creating and using a ClassificationResultItem."""
    print("=== ClassificationResultItem Example ===")
    
    # Create a classification result item
    item = ClassificationResultItem(
        label="Classified Model",
        classified_mesh=None,  # Would be an Open3D mesh object
        original_mesh_path="/path/to/original.obj",
        classification_results={},
        confidence_scores={},
        is_visible=True
    )
    
    print(f"Item ID: {item.id}")
    print(f"Label: {item.label}")
    print(f"Primary Class: {item.primary_class}")
    print(f"Primary Confidence: {item.primary_confidence}")
    
    # Add classifications
    item.add_classification("vase", 0.85, {"material": "ceramic", "era": "ancient"})
    item.add_classification("bowl", 0.72, {"material": "clay", "era": "medieval"})
    item.add_classification("pot", 0.65, {"material": "terracotta", "era": "modern"})
    
    print(f"Classification Results: {item.classification_results}")
    print(f"Confidence Scores: {item.confidence_scores}")
    print(f"Primary Class: {item.primary_class}")
    print(f"Primary Confidence: {item.primary_confidence}")
    print(f"Top 2 Classifications: {item.get_top_classifications(2)}")
    
    # Convert to dictionary
    item_dict = item.to_dict()
    print(f"Dictionary representation: {item_dict}")
    print()


def example_pairwise_item():
    """Example of creating and using a PairwiseResultItem."""
    print("=== PairwiseResultItem Example ===")
    
    # Create a pairwise result item
    transformation_matrix = np.eye(4)  # Identity matrix
    transformation_matrix[:3, 3] = [1.0, 2.0, 3.0]  # Translation
    
    item = PairwiseResultItem(
        label="Fragment Pair Match",
        fragment1_path="/path/to/fragment1.obj",
        fragment2_path="/path/to/fragment2.obj",
        matched_mesh=None,  # Would be an Open3D mesh object
        transformation_matrix=transformation_matrix,
        matching_score=0.92,
        matching_parameters={
            "method": "icp",
            "max_iterations": 1000,
            "tolerance": 1e-6
        },
        is_visible=True
    )
    
    print(f"Item ID: {item.id}")
    print(f"Label: {item.label}")
    print(f"Fragment 1: {item.fragment1_path}")
    print(f"Fragment 2: {item.fragment2_path}")
    print(f"Matching Score: {item.matching_score}")
    print(f"Transformation Matrix:\n{item.transformation_matrix}")
    print(f"Translation: {item.get_translation()}")
    print(f"Rotation Matrix:\n{item.get_rotation_matrix()}")
    
    # Add correspondence points
    point1 = np.array([0.0, 0.0, 0.0])
    point2 = np.array([1.0, 1.0, 1.0])
    item.add_correspondence_point(point1, point2)
    print(f"Correspondence Count: {item.get_correspondence_count()}")
    
    # Convert to dictionary
    item_dict = item.to_dict()
    print(f"Dictionary representation: {item_dict}")
    print()


def example_assembly_item():
    """Example of creating and using an AssemblyResultItem."""
    print("=== AssemblyResultItem Example ===")
    
    # Create an assembly result item
    item = AssemblyResultItem(
        label="Complete Assembly",
        assembled_mesh=None,  # Would be an Open3D mesh object
        fragment_paths=["/path/to/frag1.obj", "/path/to/frag2.obj", "/path/to/frag3.obj"],
        assembly_parameters={
            "method": "global_optimization",
            "max_iterations": 5000,
            "convergence_threshold": 1e-5
        },
        assembly_score=0.95,
        fragment_count=3,
        is_visible=True
    )
    
    print(f"Item ID: {item.id}")
    print(f"Label: {item.label}")
    print(f"Fragment Paths: {item.fragment_paths}")
    print(f"Fragment Count: {item.fragment_count}")
    print(f"Assembly Score: {item.assembly_score}")
    print(f"Completion Percentage: {item.completion_percentage}")
    print(f"Is Complete: {item.is_complete()}")
    
    # Add more fragments
    item.add_fragment_path("/path/to/frag4.obj")
    print(f"Updated Fragment Count: {item.fragment_count}")
    
    # Set completion percentage
    item.completion_percentage = 85.5
    print(f"Updated Completion: {item.completion_percentage}%")
    print(f"Is Complete: {item.is_complete()}")
    
    # Add assembly metadata
    item.add_assembly_metadata("assembly_time", "2.5 hours")
    item.add_assembly_metadata("algorithm_version", "v2.1")
    print(f"Assembly Metadata: {item.assembly_metadata}")
    
    # Convert to dictionary
    item_dict = item.to_dict()
    print(f"Dictionary representation: {item_dict}")
    print()


def main():
    """Run all examples."""
    print("Item System Examples")
    print("=" * 50)
    
    example_base_model_item()
    example_preprocessed_item()
    example_segmentation_item()
    example_classification_item()
    example_pairwise_item()
    example_assembly_item()
    
    print("All examples completed!")


if __name__ == "__main__":
    main() 