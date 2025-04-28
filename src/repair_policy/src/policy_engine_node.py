#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Policy Engine Node for QTrobot Confusion Detection System

This node selects and executes appropriate repair strategies in response
to detected confusion.
"""

import rospy
import yaml
import os
from confusion_detection.msg import ConfusionState
from std_msgs.msg import String
from repair_policy.srv import RepairStrategy
import repair_strategies

class PolicyEngineNode:
    """ROS node for selecting and executing repair strategies."""
    
    def __init__(self):
        """Initialize the policy engine node."""
        rospy.init_node('policy_engine_node', anonymous=True)
        
        # Parameters
        self.config_path = rospy.get_param('~config_path', 'config/repair_policies.yaml')
        self.min_confusion_score = rospy.get_param('~min_confusion_score', 0.6)
        self.min_confidence = rospy.get_param('~min_confidence', 0.7)
        self.max_repair_attempts = rospy.get_param('~max_repair_attempts', 3)
        self.repair_cooldown = rospy.get_param('~repair_cooldown', 5.0)  # seconds
        
        # Load configuration
        self.load_config()
        
        # State variables
        self.current_topic = ""
        self.conversation_history = []
        self.previous_repair_attempts = 0
        self.previous_strategies_used = []
        self.last_repair_time = 0
        self.is_repairing = False
        
        # Publishers and subscribers
        self.confusion_state_sub = rospy.Subscriber('/human/confusion_state', ConfusionState, self.confusion_callback)
        self.speech_pub = rospy.Publisher('/robot/speech/say', String, queue_size=10)
        self.behavior_pub = rospy.Publisher('/robot/behavior/trigger', String, queue_size=10)
        
        # Services
        self.repair_service = rospy.Service('/repair_policy/get_strategy', RepairStrategy, self.get_repair_strategy)
        
        rospy.loginfo("Policy engine node initialized")
    
    def load_config(self):
        """Load configuration from YAML file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                rospy.loginfo(f"Loaded configuration from {self.config_path}")
            else:
                rospy.logwarn(f"Configuration file {self.config_path} not found, using defaults")
                self.config = {}
        except Exception as e:
            rospy.logerr(f"Failed to load configuration: {e}")
            self.config = {}
    
    def confusion_callback(self, data):
        """Process incoming confusion state messages."""
        try:
            # Check if confusion is detected with sufficient confidence
            if (data.is_confused and 
                data.confusion_score >= self.min_confusion_score and 
                data.confidence >= self.min_confidence):
                
                # Check if we're in the cooldown period
                current_time = rospy.get_time()
                if current_time - self.last_repair_time < self.repair_cooldown:
                    return
                
                # Check if we've exceeded the maximum number of repair attempts
                if self.previous_repair_attempts >= self.max_repair_attempts:
                    rospy.logwarn("Maximum repair attempts exceeded, escalating to human operator")
                    self.escalate_to_human()
                    return
                
                # Select and execute a repair strategy
                self.execute_repair_strategy(data)
                
            elif not data.is_confused and self.is_repairing:
                # If confusion has been resolved, reset repair state
                self.reset_repair_state()
                
        except Exception as e:
            rospy.logerr(f"Error processing confusion state: {e}")
    
    def get_repair_strategy(self, req):
        """Service handler for getting a repair strategy."""
        try:
            # Create context for strategy selection
            context = {
                'current_topic': req.current_topic,
                'conversation_history': req.conversation_history,
                'confusion_duration': req.confusion_duration,
                'previous_repair_attempts': req.previous_repair_attempts,
                'previous_strategies_used': req.previous_strategies_used
            }
            
            # Select a repair strategy
            strategy = repair_strategies.select_strategy(req.confusion_state, context)
            
            # Generate repair message
            repair_message = strategy.generate_repair_message(req.confusion_state, context)
            
            # Create response
            response = {
                'strategy_name': strategy.name,
                'strategy_parameters': [],  # Placeholder
                'repair_message': repair_message,
                'confidence': strategy.success_rate,
                'escalate_to_human': False  # Default
            }
            
            # Check if we should escalate to human
            if (req.previous_repair_attempts >= self.max_repair_attempts or
                (req.confusion_duration > 30.0 and strategy.success_rate < 0.3)):
                response['escalate_to_human'] = True
            
            return response
            
        except Exception as e:
            rospy.logerr(f"Error selecting repair strategy: {e}")
            return {
                'strategy_name': 'fallback',
                'strategy_parameters': [],
                'repair_message': "I'm having trouble understanding. Let me get help.",
                'confidence': 0.0,
                'escalate_to_human': True
            }
    
    def execute_repair_strategy(self, confusion_state):
        """Execute a repair strategy based on the confusion state."""
        try:
            # Create context for strategy selection
            context = {
                'current_topic': self.current_topic,
                'conversation_history': self.conversation_history,
                'confusion_duration': confusion_state.duration,
                'previous_repair_attempts': self.previous_repair_attempts,
                'previous_strategies_used': self.previous_strategies_used
            }
            
            # Select a repair strategy
            strategy = repair_strategies.select_strategy(confusion_state, context)
            
            # Generate repair message
            repair_message = strategy.generate_repair_message(confusion_state, context)
            
            # Execute the repair
            self.speak(repair_message)
            
            # Update state
            self.is_repairing = True
            self.previous_repair_attempts += 1
            self.previous_strategies_used.append(strategy.name)
            self.last_repair_time = rospy.get_time()
            
            rospy.loginfo(f"Executed repair strategy: {strategy.name}")
            
        except Exception as e:
            rospy.logerr(f"Error executing repair strategy: {e}")
    
    def speak(self, text):
        """Make the robot speak the given text."""
        try:
            msg = String()
            msg.data = text
            self.speech_pub.publish(msg)
            rospy.loginfo(f"Robot says: {text}")
        except Exception as e:
            rospy.logerr(f"Error making robot speak: {e}")
    
    def trigger_behavior(self, behavior_name):
        """Trigger a predefined robot behavior."""
        try:
            msg = String()
            msg.data = behavior_name
            self.behavior_pub.publish(msg)
            rospy.loginfo(f"Triggered behavior: {behavior_name}")
        except Exception as e:
            rospy.logerr(f"Error triggering behavior: {e}")
    
    def escalate_to_human(self):
        """Escalate the situation to a human operator."""
        try:
            # Notify the user that we're getting human help
            self.speak("I'm having trouble understanding. Let me get some help.")
            
            # Trigger a behavior to indicate escalation
            self.trigger_behavior("escalate_to_human")
            
            # Reset repair state
            self.reset_repair_state()
            
            rospy.loginfo("Escalated to human operator")
            
        except Exception as e:
            rospy.logerr(f"Error escalating to human: {e}")
    
    def reset_repair_state(self):
        """Reset the repair state."""
        self.is_repairing = False
        self.previous_repair_attempts = 0
        self.previous_strategies_used = []
    
    def update_conversation_history(self, message, is_robot=True):
        """Update the conversation history with a new message."""
        # Add the message to the history
        self.conversation_history.append({
            'text': message,
            'is_robot': is_robot,
            'timestamp': rospy.get_time()
        })
        
        # Limit the history size
        max_history = 10
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
    
    def run(self):
        """Run the policy engine node."""
        rospy.spin()

if __name__ == '__main__':
    try:
        engine = PolicyEngineNode()
        engine.run()
    except rospy.ROSInterruptException:
        pass
