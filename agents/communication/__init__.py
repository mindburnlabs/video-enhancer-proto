"""
Agent Communication System Module

Provides comprehensive inter-agent communication infrastructure including:
- Message passing and queuing
- Agent registry and status monitoring
- Error handling and performance tracking
- Task coordination and workflow management
"""

from .agent_communication_system import (
    AgentCommunicationSystem,
    MessageBroker,
    AgentRegistry,
    AgentStatus,
    MessageDeliveryResult,
    MessageQueueStats,
    CommunicationStatus,
    get_communication_system,
    initialize_communication_system,
    shutdown_communication_system
)

__all__ = [
    'AgentCommunicationSystem',
    'MessageBroker',
    'AgentRegistry',
    'AgentStatus',
    'MessageDeliveryResult',
    'MessageQueueStats',
    'CommunicationStatus',
    'get_communication_system',
    'initialize_communication_system',
    'shutdown_communication_system'
]