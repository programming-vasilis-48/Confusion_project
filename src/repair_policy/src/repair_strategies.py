#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repair Strategies Module for QTrobot Confusion Detection System

This module defines various conversational repair strategies that can be
employed when confusion is detected.
"""

import rospy
import random
from enum import Enum, auto

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

class RepairStrategy:
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

class ClarificationStrategy(RepairStrategy):
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

class SimplificationStrategy(RepairStrategy):
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

class VisualSupportStrategy(RepairStrategy):
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

class EngagementCheckStrategy(RepairStrategy):
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

class TopicShiftStrategy(RepairStrategy):
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

def get_all_strategies():
    """Get all available repair strategies."""
    return REPAIR_STRATEGIES

def get_strategy(strategy_type):
    """Get a specific repair strategy by type."""
    return REPAIR_STRATEGIES.get(strategy_type)

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
