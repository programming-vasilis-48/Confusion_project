#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Utilities for QTrobot Confusion Detection System

This module provides visualization utilities for the confusion detection
and repair policy packages.
"""

import rospy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import os
from datetime import datetime, timedelta

class ConfusionVisualizer:
    """Class for visualizing confusion detection results."""
    
    def __init__(self, buffer_size=100):
        """Initialize the confusion visualizer."""
        self.buffer_size = buffer_size
        self.timestamps = []
        self.confusion_scores = []
        self.is_confused = []
        self.confidence_values = []
        self.repair_events = []
    
    def update(self, timestamp, confusion_score, is_confused, confidence):
        """Update the visualization data."""
        # Add new data point
        self.timestamps.append(timestamp)
        self.confusion_scores.append(confusion_score)
        self.is_confused.append(1 if is_confused else 0)
        self.confidence_values.append(confidence)
        
        # Trim buffers if they exceed the buffer size
        if len(self.timestamps) > self.buffer_size:
            self.timestamps = self.timestamps[-self.buffer_size:]
            self.confusion_scores = self.confusion_scores[-self.buffer_size:]
            self.is_confused = self.is_confused[-self.buffer_size:]
            self.confidence_values = self.confidence_values[-self.buffer_size:]
    
    def add_repair_event(self, timestamp, strategy_name):
        """Add a repair event to the visualization."""
        self.repair_events.append((timestamp, strategy_name))
        
        # Trim repair events buffer if it exceeds the buffer size
        if len(self.repair_events) > self.buffer_size:
            self.repair_events = self.repair_events[-self.buffer_size:]
    
    def plot_confusion_timeline(self, save_path=None):
        """Plot the confusion timeline."""
        try:
            # Create figure and axes
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Plot confusion score
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Confusion Score', color='tab:blue')
            ax1.plot(self.timestamps, self.confusion_scores, 'b-', label='Confusion Score')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.set_ylim(0, 1)
            
            # Create second y-axis for confidence
            ax2 = ax1.twinx()
            ax2.set_ylabel('Confidence', color='tab:orange')
            ax2.plot(self.timestamps, self.confidence_values, 'r-', label='Confidence')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            ax2.set_ylim(0, 1)
            
            # Highlight confused regions
            for i, confused in enumerate(self.is_confused):
                if confused and i > 0:
                    ax1.axvspan(self.timestamps[i-1], self.timestamps[i], alpha=0.3, color='red')
            
            # Mark repair events
            for timestamp, strategy_name in self.repair_events:
                ax1.axvline(x=timestamp, color='g', linestyle='--', alpha=0.7)
                ax1.text(timestamp, 0.5, strategy_name, rotation=90, verticalalignment='center')
            
            # Set title and legend
            plt.title('Confusion Detection Timeline')
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.tight_layout()
            
            # Save or display the plot
            if save_path:
                plt.savefig(save_path)
                rospy.loginfo(f"Saved confusion timeline plot to {save_path}")
                plt.close(fig)
            else:
                return fig
                
        except Exception as e:
            rospy.logerr(f"Failed to plot confusion timeline: {e}")
            return None
    
    def create_confusion_heatmap(self, confusion_matrix, labels, save_path=None):
        """Create a heatmap visualization of the confusion matrix."""
        try:
            # Create figure and axes
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create heatmap
            im = ax.imshow(confusion_matrix, cmap='Blues')
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            
            # Rotate x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, f"{confusion_matrix[i, j]:.2f}",
                            ha="center", va="center", color="black" if confusion_matrix[i, j] < 0.5 else "white")
            
            # Set title and labels
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            
            plt.tight_layout()
            
            # Save or display the plot
            if save_path:
                plt.savefig(save_path)
                rospy.loginfo(f"Saved confusion heatmap to {save_path}")
                plt.close(fig)
            else:
                return fig
                
        except Exception as e:
            rospy.logerr(f"Failed to create confusion heatmap: {e}")
            return None
    
    def create_repair_strategy_performance_chart(self, strategies, success_rates, usage_counts, save_path=None):
        """Create a bar chart of repair strategy performance."""
        try:
            # Create figure and axes
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Set width of bars
            bar_width = 0.35
            x = np.arange(len(strategies))
            
            # Plot success rates
            ax1.set_xlabel('Repair Strategy')
            ax1.set_ylabel('Success Rate', color='tab:blue')
            bars1 = ax1.bar(x - bar_width/2, success_rates, bar_width, label='Success Rate', color='tab:blue')
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.set_ylim(0, 1)
            
            # Create second y-axis for usage counts
            ax2 = ax1.twinx()
            ax2.set_ylabel('Usage Count', color='tab:orange')
            bars2 = ax2.bar(x + bar_width/2, usage_counts, bar_width, label='Usage Count', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            
            # Set x-ticks and labels
            ax1.set_xticks(x)
            ax1.set_xticklabels(strategies, rotation=45, ha='right')
            
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{height:.2f}', ha='center', va='bottom')
            
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{int(height)}', ha='center', va='bottom')
            
            # Set title and legend
            plt.title('Repair Strategy Performance')
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            plt.tight_layout()
            
            # Save or display the plot
            if save_path:
                plt.savefig(save_path)
                rospy.loginfo(f"Saved repair strategy performance chart to {save_path}")
                plt.close(fig)
            else:
                return fig
                
        except Exception as e:
            rospy.logerr(f"Failed to create repair strategy performance chart: {e}")
            return None
    
    def create_reaction_time_histogram(self, reaction_times, bins=10, save_path=None):
        """Create a histogram of reaction times."""
        try:
            # Create figure and axes
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Create histogram
            n, bins, patches = ax.hist(reaction_times, bins=bins, alpha=0.7, color='tab:blue')
            
            # Add mean line
            mean_reaction_time = np.mean(reaction_times)
            ax.axvline(x=mean_reaction_time, color='r', linestyle='--', label=f'Mean: {mean_reaction_time:.2f}s')
            
            # Set title and labels
            ax.set_title("Reaction Time Histogram")
            ax.set_xlabel("Reaction Time (seconds)")
            ax.set_ylabel("Frequency")
            ax.legend()
            
            plt.tight_layout()
            
            # Save or display the plot
            if save_path:
                plt.savefig(save_path)
                rospy.loginfo(f"Saved reaction time histogram to {save_path}")
                plt.close(fig)
            else:
                return fig
                
        except Exception as e:
            rospy.logerr(f"Failed to create reaction time histogram: {e}")
            return None
    
    def create_confusion_overlay(self, image, confusion_score, is_confused, confidence):
        """Create an image with confusion information overlay."""
        try:
            # Create a copy of the image
            overlay = image.copy()
            
            # Add confusion information as text overlay
            height, width = overlay.shape[:2]
            
            # Create a semi-transparent rectangle for the text background
            cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, image, 0.3, 0, overlay, (10, 10, 300, 120))
            
            # Add text with confusion information
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            font_color = (255, 255, 255)
            
            # Confusion score
            cv2.putText(overlay, f"Confusion Score: {confusion_score:.2f}", 
                       (20, 40), font, font_scale, font_color, font_thickness)
            
            # Confusion state
            state_text = "CONFUSED" if is_confused else "NOT CONFUSED"
            state_color = (0, 0, 255) if is_confused else (0, 255, 0)
            cv2.putText(overlay, f"State: {state_text}", 
                       (20, 70), font, font_scale, state_color, font_thickness)
            
            # Confidence
            cv2.putText(overlay, f"Confidence: {confidence:.2f}", 
                       (20, 100), font, font_scale, font_color, font_thickness)
            
            return overlay
            
        except Exception as e:
            rospy.logerr(f"Failed to create confusion overlay: {e}")
            return image
