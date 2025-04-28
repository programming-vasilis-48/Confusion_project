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
import random
from enum import Enum, auto
from confusion_detection.msg import ConfusionState
from std_msgs.msg import String
from repair_policy.srv import RepairStrategy, RepairStrategyResponse

# Define repair strategy types
class RepairStrategyType(Enum):
    """Enumeration of repair strategy types."""
    CLARIFICATION = auto()
    SIMPLIFICATION = auto()
    VISUAL_SUPPORT = auto()
    ENGAGEMENT_CHECK = auto()
    TOPIC_SHIFT = auto()
    REPETITION = auto()
    ELABORATION = auto()
    EXAMPLE = auto()
    SUMMARY = auto()
    PAUSE = auto()

class RepairStrategyClass:
    """Base class for repair strategies."""

    def __init__(self, name, description, priority=1.0):
        """Initialize the repair strategy."""
        self.name = name
        self.description = description
        self.priority = priority  # Higher priority strategies are preferred
        self.success_rate = 0.5  # Initial success rate estimate
        self.usage_count = 0
        self.success_count = 0

    def is_applicable(self, confusion_state, context):
        """Check if the strategy is applicable in the current context."""
        # Base implementation always returns True
        # Subclasses should override this method with specific logic
        return True

    def generate_repair_message(self, confusion_state, context):
        """Generate a repair message based on the confusion state and context."""
        # Base implementation returns a generic message
        # Subclasses should override this method with specific logic
        return "I notice you might be confused. Let me try to help."

    def update_success_rate(self, success):
        """Update the success rate of the strategy."""
        self.usage_count += 1
        if success:
            self.success_count += 1
        self.success_rate = self.success_count / self.usage_count if self.usage_count > 0 else 0.5

class ClarificationStrategy(RepairStrategyClass):
    """Strategy that clarifies the previous statement."""

    def __init__(self):
        """Initialize the clarification strategy."""
        super().__init__(
            name="clarification",
            description="Clarifies the previous statement",
            priority=0.8
        )

    def is_applicable(self, confusion_state, context):
        """Check if clarification is applicable."""
        # Clarification is applicable if there's a previous statement to clarify
        return len(context.get('conversation_history', [])) > 0

    def generate_repair_message(self, confusion_state, context):
        """Generate a clarification message."""
        templates = [
            "Let me clarify what I meant. {}",
            "To be more clear, {}",
            "What I'm trying to say is {}",
            "Let me rephrase that. {}",
            "In other words, {}"
        ]

        # In a real implementation, this would use NLG techniques to generate
        # a clarification of the previous statement based on the context

        # For now, we'll just use a template
        template = random.choice(templates)
        clarification = "I'll explain this differently."  # Placeholder

        return template.format(clarification)

class SimplificationStrategy(RepairStrategyClass):
    """Strategy that simplifies the previous explanation."""

    def __init__(self):
        """Initialize the simplification strategy."""
        super().__init__(
            name="simplification",
            description="Simplifies the previous explanation",
            priority=0.7
        )

    def is_applicable(self, confusion_state, context):
        """Check if simplification is applicable."""
        # Simplification is applicable if there's a previous explanation to simplify
        return len(context.get('conversation_history', [])) > 0

    def generate_repair_message(self, confusion_state, context):
        """Generate a simplified explanation."""
        templates = [
            "Let me simplify this. {}",
            "To put it simply, {}",
            "In simpler terms, {}",
            "The basic idea is {}",
            "To break it down, {}"
        ]

        # In a real implementation, this would use NLG techniques to generate
        # a simplified version of the previous explanation

        # For now, we'll just use a template
        template = random.choice(templates)
        simplification = "Let's focus on the main point."  # Placeholder

        return template.format(simplification)

class VisualSupportStrategy(RepairStrategyClass):
    """Strategy that provides visual support for the explanation."""

    def __init__(self):
        """Initialize the visual support strategy."""
        super().__init__(
            name="visual_support",
            description="Provides visual support for the explanation",
            priority=0.6
        )

    def is_applicable(self, confusion_state, context):
        """Check if visual support is applicable."""
        # Visual support is applicable if there's a visual aid available for the current topic
        # In a real implementation, this would check if there's a relevant image or animation
        return True  # Placeholder

    def generate_repair_message(self, confusion_state, context):
        """Generate a message with visual support."""
        templates = [
            "Let me show you a visual to help explain. {}",
            "Here's an image that might help. {}",
            "Sometimes a picture helps. {}",
            "Let me illustrate this. {}",
            "This visual should make it clearer. {}"
        ]

        # In a real implementation, this would trigger the display of a relevant
        # image or animation on the robot's screen

        # For now, we'll just use a template
        template = random.choice(templates)
        visual_description = "I'm displaying a helpful diagram now."  # Placeholder

        return template.format(visual_description)

class EngagementCheckStrategy(RepairStrategyClass):
    """Strategy that checks if the user is still engaged and understanding."""

    def __init__(self):
        """Initialize the engagement check strategy."""
        super().__init__(
            name="engagement_check",
            description="Checks if the user is still engaged and understanding",
            priority=0.5
        )

    def is_applicable(self, confusion_state, context):
        """Check if engagement check is applicable."""
        # Engagement check is always applicable
        return True

    def generate_repair_message(self, confusion_state, context):
        """Generate an engagement check message."""
        templates = [
            "Does that make sense to you?",
            "Are you following me so far?",
            "Is this clear, or should I explain differently?",
            "How does that sound to you?",
            "Do you understand, or would you like me to clarify?"
        ]

        # Simply choose a random template
        return random.choice(templates)

class TopicShiftStrategy(RepairStrategyClass):
    """Strategy that shifts to a different topic when persistent confusion is detected."""

    def __init__(self):
        """Initialize the topic shift strategy."""
        super().__init__(
            name="topic_shift",
            description="Shifts to a different topic when persistent confusion is detected",
            priority=0.3  # Lower priority as this is a more drastic strategy
        )

    def is_applicable(self, confusion_state, context):
        """Check if topic shift is applicable."""
        # Topic shift is applicable if confusion has persisted for a while
        # and multiple repair attempts have failed
        confusion_duration = context.get('confusion_duration', 0.0)
        previous_attempts = context.get('previous_repair_attempts', 0)
        return confusion_duration > 10.0 or previous_attempts >= 3

    def generate_repair_message(self, confusion_state, context):
        """Generate a topic shift message."""
        templates = [
            "Let's try a different approach. {}",
            "Maybe we should move on to something else. {}",
            "Let's switch gears for a moment. {}",
            "I think we should try a different topic. {}",
            "Let's take a step back and look at this differently. {}"
        ]

        # In a real implementation, this would select a related but different
        # topic based on the conversation context

        # For now, we'll just use a template
        template = random.choice(templates)
        new_topic = "Would you like to discuss something else instead?"  # Placeholder

        return template.format(new_topic)

# Dictionary of available repair strategies
REPAIR_STRATEGIES = {
    RepairStrategyType.CLARIFICATION: ClarificationStrategy(),
    RepairStrategyType.SIMPLIFICATION: SimplificationStrategy(),
    RepairStrategyType.VISUAL_SUPPORT: VisualSupportStrategy(),
    RepairStrategyType.ENGAGEMENT_CHECK: EngagementCheckStrategy(),
    RepairStrategyType.TOPIC_SHIFT: TopicShiftStrategy(),
}

def select_strategy(confusion_state, context):
    """Select the most appropriate repair strategy based on the confusion state and context."""
    applicable_strategies = []

    # Find all applicable strategies
    for strategy_type, strategy in REPAIR_STRATEGIES.items():
        if strategy.is_applicable(confusion_state, context):
            applicable_strategies.append((strategy_type, strategy))

    if not applicable_strategies:
        # If no strategies are applicable, default to engagement check
        return REPAIR_STRATEGIES[RepairStrategyType.ENGAGEMENT_CHECK]

    # Sort by priority and success rate
    applicable_strategies.sort(
        key=lambda x: (x[1].priority, x[1].success_rate),
        reverse=True
    )

    # Avoid using the same strategy repeatedly
    previous_strategies = context.get('previous_strategies_used', [])
    for strategy_type, strategy in applicable_strategies:
        if strategy.name not in previous_strategies:
            return strategy

    # If all strategies have been used, return the highest priority one
    return applicable_strategies[0][1]

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
            strategy = select_strategy(req.confusion_state, context)

            # Generate repair message
            repair_message = strategy.generate_repair_message(req.confusion_state, context)

            # Create response
            response = RepairStrategyResponse()
            response.strategy_name = strategy.name
            response.strategy_parameters = []  # Placeholder
            response.repair_message = repair_message
            response.confidence = strategy.success_rate
            response.escalate_to_human = False  # Default

            # Check if we should escalate to human
            if (req.previous_repair_attempts >= self.max_repair_attempts or
                (req.confusion_duration > 30.0 and strategy.success_rate < 0.3)):
                response.escalate_to_human = True

            return response

        except Exception as e:
            rospy.logerr(f"Error selecting repair strategy: {e}")
            response = RepairStrategyResponse()
            response.strategy_name = 'fallback'
            response.strategy_parameters = []
            response.repair_message = "I'm having trouble understanding. Let me get help."
            response.confidence = 0.0
            response.escalate_to_human = True
            return response

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
            strategy = select_strategy(confusion_state, context)

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
