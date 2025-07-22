# Item System for Reassembly Application

This directory contains the implementation of a structured item system for the Reassembly application's item tree.

## Overview

The item system provides a hierarchical structure for managing different types of items with their specific properties and behaviors.

## Item Types

### BaseItem (Abstract Base Class)
- **Purpose**: Defines common properties and methods for all items
- **Properties**: id, label, is_visible, ui_widget

### BaseModelItem
- **Purpose**: Represents original 3D models/fragments
- **Properties**: mesh_path, mesh, plus base properties

### PreprocessedItem
- **Purpose**: Represents preprocessing results
- **Properties**: preprocessed_mesh, original_mesh_path, preprocessing_parameters, preprocessing_steps, quality_metrics, processing_time, mesh_quality_score

### SegmentationResultItem
- **Purpose**: Represents mesh segmentation results
- **Properties**: segmented_mesh, original_mesh_path, segmentation_parameters, segment_count, segment_labels

### ClassificationResultItem
- **Purpose**: Represents model classification results
- **Properties**: classified_mesh, classification_results, confidence_scores, primary_class

### PairwiseResultItem
- **Purpose**: Represents pairwise matching results
- **Properties**: fragment1_path, fragment2_path, matched_mesh, transformation_matrix, matching_score

### AssemblyResultItem
- **Purpose**: Represents final assembly results
- **Properties**: assembled_mesh, fragment_paths, assembly_score, completion_percentage

## Usage

See `example_item_usage.py` for detailed usage examples. 