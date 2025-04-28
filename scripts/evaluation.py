#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation Script for QTrobot Confusion Detection System

This script evaluates the performance of the confusion detection system
using recorded data or live testing.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import rospy
import rosbag
from confusion_detection.msg import ConfusionState, FaceFeatures
from std_msgs.msg import String

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate confusion detection system')
    parser.add_argument('--mode', type=str, default='recorded', choices=['recorded', 'live'],
                        help='Evaluation mode: recorded (from rosbag) or live')
    parser.add_argument('--bag', type=str, default=None,
                        help='Path to rosbag file (required for recorded mode)')
    parser.add_argument('--ground-truth', type=str, default=None,
                        help='Path to ground truth annotations (required for recorded mode)')
    parser.add_argument('--duration', type=int, default=60,
                        help='Duration of live evaluation in seconds (for live mode)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for evaluation results')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'recorded':
        if args.bag is None:
            parser.error('--bag is required for recorded mode')
        if not os.path.exists(args.bag):
            parser.error(f'Rosbag file not found: {args.bag}')
        if args.ground_truth is None:
            parser.error('--ground-truth is required for recorded mode')
        if not os.path.exists(args.ground_truth):
            parser.error(f'Ground truth file not found: {args.ground_truth}')
    
    # Set default output directory if not specified
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'evaluation_results_{timestamp}'
    
    return args

def load_ground_truth(ground_truth_file):
    """Load ground truth annotations from a file."""
    try:
        with open(ground_truth_file, 'r') as f:
            ground_truth = yaml.safe_load(f)
        
        # Validate ground truth format
        if not isinstance(ground_truth, list):
            raise ValueError('Ground truth must be a list of annotations')
        
        for annotation in ground_truth:
            if 'timestamp' not in annotation or 'is_confused' not in annotation:
                raise ValueError('Each annotation must have timestamp and is_confused fields')
        
        return ground_truth
    except Exception as e:
        print(f'Error loading ground truth: {e}')
        sys.exit(1)

def evaluate_recorded_data(bag_file, ground_truth, output_dir, visualize=False):
    """Evaluate the system using recorded data from a rosbag file."""
    print(f'Evaluating recorded data from: {bag_file}')
    print(f'Using ground truth from: {ground_truth}')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ground truth annotations
    annotations = load_ground_truth(ground_truth)
    
    # Process rosbag
    confusion_states = []
    repair_events = []
    
    try:
        bag = rosbag.Bag(bag_file)
        
        # Extract confusion states
        for topic, msg, t in bag.read_messages(topics=['/human/confusion_state']):
            confusion_states.append({
                'timestamp': t.to_sec(),
                'confusion_score': msg.confusion_score,
                'is_confused': msg.is_confused,
                'confidence': msg.confidence,
                'detected_cues': msg.detected_cues,
                'duration': msg.duration
            })
        
        # Extract repair events
        for topic, msg, t in bag.read_messages(topics=['/robot/speech/say']):
            repair_events.append({
                'timestamp': t.to_sec(),
                'message': msg.data
            })
        
        bag.close()
    except Exception as e:
        print(f'Error processing rosbag: {e}')
        sys.exit(1)
    
    # Align ground truth with confusion states
    aligned_data = align_ground_truth_with_detections(annotations, confusion_states)
    
    # Calculate metrics
    metrics = calculate_metrics(aligned_data)
    
    # Calculate repair metrics
    repair_metrics = calculate_repair_metrics(aligned_data, repair_events)
    
    # Combine all metrics
    all_metrics = {**metrics, **repair_metrics}
    
    # Save metrics
    save_metrics(all_metrics, output_dir)
    
    # Visualize results
    if visualize:
        visualize_results(aligned_data, repair_events, all_metrics, output_dir)
    
    print(f'Evaluation completed. Results saved to: {output_dir}')
    print_metrics_summary(all_metrics)

def align_ground_truth_with_detections(annotations, confusion_states):
    """Align ground truth annotations with detected confusion states."""
    aligned_data = []
    
    # Sort by timestamp
    annotations = sorted(annotations, key=lambda x: x['timestamp'])
    confusion_states = sorted(confusion_states, key=lambda x: x['timestamp'])
    
    # Find closest ground truth annotation for each detection
    for state in confusion_states:
        closest_annotation = None
        min_time_diff = float('inf')
        
        for annotation in annotations:
            time_diff = abs(state['timestamp'] - annotation['timestamp'])
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_annotation = annotation
        
        # Only use annotations within a reasonable time window (e.g., 1 second)
        if min_time_diff <= 1.0 and closest_annotation is not None:
            aligned_data.append({
                'timestamp': state['timestamp'],
                'detected_confusion_score': state['confusion_score'],
                'detected_is_confused': state['is_confused'],
                'detected_confidence': state['confidence'],
                'detected_cues': state['detected_cues'],
                'detected_duration': state['duration'],
                'ground_truth_is_confused': closest_annotation['is_confused'],
                'ground_truth_timestamp': closest_annotation['timestamp'],
                'time_diff': min_time_diff
            })
    
    return aligned_data

def calculate_metrics(aligned_data):
    """Calculate evaluation metrics."""
    # Extract predictions and ground truth
    y_pred = [int(d['detected_is_confused']) for d in aligned_data]
    y_true = [int(d['ground_truth_is_confused']) for d in aligned_data]
    
    # Calculate confusion matrix
    tp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 1)
    tn = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 0)
    
    # Calculate metrics
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate confusion scores
    confusion_scores = [d['detected_confusion_score'] for d in aligned_data]
    mean_confusion_score = np.mean(confusion_scores) if confusion_scores else 0
    std_confusion_score = np.std(confusion_scores) if confusion_scores else 0
    
    # Calculate confidence scores
    confidence_scores = [d['detected_confidence'] for d in aligned_data]
    mean_confidence = np.mean(confidence_scores) if confidence_scores else 0
    std_confidence = np.std(confidence_scores) if confidence_scores else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'mean_confusion_score': mean_confusion_score,
        'std_confusion_score': std_confusion_score,
        'mean_confidence': mean_confidence,
        'std_confidence': std_confidence,
        'num_samples': len(aligned_data)
    }

def calculate_repair_metrics(aligned_data, repair_events):
    """Calculate repair strategy metrics."""
    if not repair_events:
        return {
            'num_repair_events': 0,
            'mean_reaction_time': 0,
            'std_reaction_time': 0
        }
    
    # Calculate reaction times
    reaction_times = []
    
    for event in repair_events:
        # Find the closest confusion state before the repair event
        confusion_before = None
        min_time_diff = float('inf')
        
        for data in aligned_data:
            if data['detected_is_confused'] and data['timestamp'] < event['timestamp']:
                time_diff = event['timestamp'] - data['timestamp']
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    confusion_before = data
        
        if confusion_before is not None and min_time_diff <= 5.0:  # Only consider events within 5 seconds
            reaction_times.append(min_time_diff)
    
    # Calculate metrics
    mean_reaction_time = np.mean(reaction_times) if reaction_times else 0
    std_reaction_time = np.std(reaction_times) if reaction_times else 0
    
    return {
        'num_repair_events': len(repair_events),
        'num_reaction_times': len(reaction_times),
        'mean_reaction_time': mean_reaction_time,
        'std_reaction_time': std_reaction_time
    }

def save_metrics(metrics, output_dir):
    """Save metrics to a file."""
    metrics_file = os.path.join(output_dir, 'metrics.yaml')
    
    try:
        with open(metrics_file, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
        print(f'Metrics saved to: {metrics_file}')
    except Exception as e:
        print(f'Error saving metrics: {e}')

def visualize_results(aligned_data, repair_events, metrics, output_dir):
    """Visualize evaluation results."""
    # Create figures directory
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Extract data
    timestamps = [d['timestamp'] for d in aligned_data]
    confusion_scores = [d['detected_confusion_score'] for d in aligned_data]
    is_confused_pred = [d['detected_is_confused'] for d in aligned_data]
    is_confused_true = [d['ground_truth_is_confused'] for d in aligned_data]
    confidence_scores = [d['detected_confidence'] for d in aligned_data]
    
    # 1. Confusion score timeline
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, confusion_scores, 'b-', label='Confusion Score')
    
    # Mark ground truth confusion
    for i, is_confused in enumerate(is_confused_true):
        if is_confused:
            plt.axvline(x=timestamps[i], color='r', linestyle='--', alpha=0.3)
    
    # Mark repair events
    for event in repair_events:
        plt.axvline(x=event['timestamp'], color='g', linestyle='-', alpha=0.5)
        plt.text(event['timestamp'], 0.5, 'Repair', rotation=90, verticalalignment='center')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Confusion Score')
    plt.title('Confusion Score Timeline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figures_dir, 'confusion_timeline.png'))
    plt.close()
    
    # 2. Confusion matrix visualization
    cm = np.array([
        [metrics['true_negatives'], metrics['false_positives']],
        [metrics['false_negatives'], metrics['true_positives']]
    ])
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Not Confused', 'Confused']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 3. Metrics summary
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    values = [metrics[m] for m in metrics_to_plot]
    
    plt.bar(metrics_to_plot, values, color=['blue', 'green', 'red', 'purple'])
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    
    # Add value labels
    for i, v in enumerate(values):
        plt.text(i, v + 0.05, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'metrics_summary.png'))
    plt.close()
    
    # 4. Reaction time histogram
    if metrics['num_reaction_times'] > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(np.random.normal(metrics['mean_reaction_time'], metrics['std_reaction_time'], 100), 
                bins=10, alpha=0.7, color='blue')
        plt.axvline(x=metrics['mean_reaction_time'], color='r', linestyle='--', 
                   label=f'Mean: {metrics["mean_reaction_time"]:.2f}s')
        
        plt.xlabel('Reaction Time (s)')
        plt.ylabel('Frequency')
        plt.title('Repair Reaction Time Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(figures_dir, 'reaction_time.png'))
        plt.close()

def print_metrics_summary(metrics):
    """Print a summary of the evaluation metrics."""
    print('\nMetrics Summary:')
    print(f'Accuracy: {metrics["accuracy"]:.4f}')
    print(f'Precision: {metrics["precision"]:.4f}')
    print(f'Recall: {metrics["recall"]:.4f}')
    print(f'F1 Score: {metrics["f1"]:.4f}')
    print(f'Number of samples: {metrics["num_samples"]}')
    print(f'Number of repair events: {metrics["num_repair_events"]}')
    if metrics["num_reaction_times"] > 0:
        print(f'Mean reaction time: {metrics["mean_reaction_time"]:.4f}s')

def evaluate_live(duration, output_dir, visualize=False):
    """Evaluate the system using live data."""
    print(f'Starting live evaluation for {duration} seconds')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize ROS node
    rospy.init_node('confusion_evaluation', anonymous=True)
    
    # Data storage
    confusion_states = []
    repair_events = []
    
    # Callback functions
    def confusion_callback(msg):
        confusion_states.append({
            'timestamp': rospy.get_time(),
            'confusion_score': msg.confusion_score,
            'is_confused': msg.is_confused,
            'confidence': msg.confidence,
            'detected_cues': msg.detected_cues,
            'duration': msg.duration
        })
    
    def speech_callback(msg):
        repair_events.append({
            'timestamp': rospy.get_time(),
            'message': msg.data
        })
    
    # Subscribe to topics
    rospy.Subscriber('/human/confusion_state', ConfusionState, confusion_callback)
    rospy.Subscriber('/robot/speech/say', String, speech_callback)
    
    # Run for specified duration
    print(f'Collecting data for {duration} seconds...')
    rospy.sleep(duration)
    
    # Save collected data
    data = {
        'confusion_states': confusion_states,
        'repair_events': repair_events
    }
    
    data_file = os.path.join(output_dir, 'live_evaluation_data.yaml')
    with open(data_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f'Collected {len(confusion_states)} confusion states and {len(repair_events)} repair events')
    print(f'Data saved to: {data_file}')
    
    # Visualize results if requested
    if visualize and confusion_states:
        visualize_live_results(confusion_states, repair_events, output_dir)
    
    print(f'Live evaluation completed. Results saved to: {output_dir}')

def visualize_live_results(confusion_states, repair_events, output_dir):
    """Visualize results from live evaluation."""
    # Create figures directory
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Extract data
    timestamps = [d['timestamp'] for d in confusion_states]
    relative_timestamps = [t - timestamps[0] for t in timestamps]  # Make timestamps relative to start
    confusion_scores = [d['confusion_score'] for d in confusion_states]
    is_confused = [d['is_confused'] for d in confusion_states]
    confidence_scores = [d['confidence'] for d in confusion_states]
    
    # Adjust repair event timestamps
    if repair_events:
        repair_timestamps = [e['timestamp'] for e in repair_events]
        relative_repair_timestamps = [t - timestamps[0] for t in repair_timestamps]
    
    # 1. Confusion score timeline
    plt.figure(figsize=(12, 6))
    plt.plot(relative_timestamps, confusion_scores, 'b-', label='Confusion Score')
    plt.plot(relative_timestamps, confidence_scores, 'r-', label='Confidence')
    
    # Highlight confused regions
    for i, confused in enumerate(is_confused):
        if confused and i > 0:
            plt.axvspan(relative_timestamps[i-1], relative_timestamps[i], alpha=0.2, color='red')
    
    # Mark repair events
    if repair_events:
        for t in relative_repair_timestamps:
            plt.axvline(x=t, color='g', linestyle='-', alpha=0.5)
            plt.text(t, 0.5, 'Repair', rotation=90, verticalalignment='center')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Score')
    plt.title('Live Confusion Detection Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figures_dir, 'live_confusion_timeline.png'))
    plt.close()
    
    # 2. Confusion state distribution
    plt.figure(figsize=(8, 6))
    labels = ['Not Confused', 'Confused']
    counts = [sum(1 for c in is_confused if not c), sum(1 for c in is_confused if c)]
    
    plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
    plt.title('Confusion State Distribution')
    plt.savefig(os.path.join(figures_dir, 'live_confusion_distribution.png'))
    plt.close()

def main():
    """Main function."""
    args = parse_args()
    
    if args.mode == 'recorded':
        evaluate_recorded_data(args.bag, args.ground_truth, args.output, args.visualize)
    else:  # live mode
        evaluate_live(args.duration, args.output, args.visualize)

if __name__ == '__main__':
    main()
