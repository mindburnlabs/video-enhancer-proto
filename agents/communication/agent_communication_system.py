#!/usr/bin/env python3
"""
Agent Communication System - Multi-Agent Video Processing Pipeline

Provides robust inter-agent communication with:
- Message passing and queuing
- Status reporting and monitoring  
- Error propagation and handling
- Task coordination and synchronization
- Performance tracking and optimization
"""

"""
MIT License

Copyright (c) 2024 Video Enhancement Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import asyncio
import logging
import time
import json
import uuid
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import weakref

from agents.core import (
    AgentMessage, MessageType, TaskStatus, Priority, 
    TaskSpecification, ProcessingResult
)

logger = logging.getLogger(__name__)

class CommunicationStatus(Enum):
    """Communication system status"""
    ACTIVE = "active"
    PAUSED = "paused"
    SHUTDOWN = "shutdown"
    ERROR = "error"

@dataclass
class MessageDeliveryResult:
    """Result of message delivery attempt"""
    message_id: str
    success: bool
    delivery_time: float = 0.0
    error_message: Optional[str] = None
    retry_count: int = 0
    agent_response: Optional[Any] = None

@dataclass
class MessageQueueStats:
    """Statistics for message queue performance"""
    total_messages: int = 0
    delivered_messages: int = 0
    failed_messages: int = 0
    average_delivery_time: float = 0.0
    queue_size: int = 0
    retry_attempts: int = 0

@dataclass
class AgentStatus:
    """Status information for an agent"""
    agent_id: str
    agent_type: str
    status: str = "active"
    last_heartbeat: float = field(default_factory=time.time)
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_response_time: float = 0.0
    capabilities: Optional[Dict[str, Any]] = None

class MessageBroker:
    """Centralized message broker for agent communication"""
    
    def __init__(self, max_queue_size: int = 10000, max_workers: int = 4):
        self.max_queue_size = max_queue_size
        self.max_workers = max_workers
        
        # Message queues per agent
        self.agent_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_queue_size))
        
        # Message delivery tracking
        self.pending_messages: Dict[str, AgentMessage] = {}
        self.delivery_results: Dict[str, MessageDeliveryResult] = {}
        
        # Statistics
        self.queue_stats: Dict[str, MessageQueueStats] = defaultdict(MessageQueueStats)
        self.global_stats = MessageQueueStats()
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running = True
        self._message_processor_task = None
        self._lock = threading.Lock()
        
        logger.info("Message Broker initialized")
    
    async def start(self):
        """Start the message broker"""
        if self._message_processor_task is None:
            self._running = True
            self._message_processor_task = asyncio.create_task(self._process_messages())
            logger.info("Message Broker started")
    
    async def stop(self):
        """Stop the message broker"""
        self._running = False
        if self._message_processor_task:
            self._message_processor_task.cancel()
            try:
                await self._message_processor_task
            except asyncio.CancelledError:
                pass
        self.executor.shutdown(wait=True)
        logger.info("Message Broker stopped")
    
    async def send_message(self, message: AgentMessage, timeout: float = 30.0) -> MessageDeliveryResult:
        """Send a message to an agent"""
        start_time = time.time()
        
        # Track message
        self.pending_messages[message.id] = message
        
        # Add to recipient's queue
        receiver_id = message.receiver
        with self._lock:
            self.agent_queues[receiver_id].append(message)
            self.global_stats.total_messages += 1
            self.queue_stats[receiver_id].total_messages += 1
        
        # Wait for processing or timeout
        delivery_result = await self._wait_for_delivery(message.id, timeout)
        delivery_result.delivery_time = time.time() - start_time
        
        # Update statistics
        self._update_delivery_stats(receiver_id, delivery_result)
        
        return delivery_result
    
    async def send_message_to_multiple(self, message: AgentMessage, recipients: List[str]) -> List[MessageDeliveryResult]:
        """Send a message to multiple agents"""
        tasks = []
        for recipient in recipients:
            recipient_message = AgentMessage(
                id=str(uuid.uuid4()),
                sender=message.sender,
                receiver=recipient,
                message_type=message.message_type,
                payload=message.payload.copy(),
                priority=message.priority,
                correlation_id=message.correlation_id
            )
            tasks.append(self.send_message(recipient_message))
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def broadcast_message(self, message: AgentMessage, agent_type: str = None) -> List[MessageDeliveryResult]:
        """Broadcast a message to all agents or agents of specific type"""
        # This would need integration with agent registry
        # For now, return empty list
        logger.warning("Broadcast functionality requires agent registry integration")
        return []
    
    async def _process_messages(self):
        """Process messages in queues"""
        while self._running:
            try:
                # Process messages for each agent
                for agent_id in list(self.agent_queues.keys()):
                    await self._process_agent_queue(agent_id)
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in message processing: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_agent_queue(self, agent_id: str):
        """Process messages for a specific agent"""
        with self._lock:
            if not self.agent_queues[agent_id]:
                return
            
            message = self.agent_queues[agent_id].popleft()
        
        # Simulate message delivery (in real implementation, would send to actual agent)
        await self._deliver_message_to_agent(agent_id, message)
    
    async def _deliver_message_to_agent(self, agent_id: str, message: AgentMessage):
        """Deliver a message to an agent"""
        try:
            # Simulate delivery time
            await asyncio.sleep(0.001)
            
            # Create successful delivery result
            result = MessageDeliveryResult(
                message_id=message.id,
                success=True,
                error_message=None,
                agent_response={"status": "received", "agent_id": agent_id}
            )
            
            self.delivery_results[message.id] = result
            
        except Exception as e:
            # Create failed delivery result
            result = MessageDeliveryResult(
                message_id=message.id,
                success=False,
                error_message=str(e)
            )
            
            self.delivery_results[message.id] = result
    
    async def _wait_for_delivery(self, message_id: str, timeout: float) -> MessageDeliveryResult:
        """Wait for message delivery with timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if message_id in self.delivery_results:
                result = self.delivery_results.pop(message_id)
                if message_id in self.pending_messages:
                    del self.pending_messages[message_id]
                return result
            
            await asyncio.sleep(0.01)
        
        # Timeout occurred
        if message_id in self.pending_messages:
            del self.pending_messages[message_id]
        
        return MessageDeliveryResult(
            message_id=message_id,
            success=False,
            error_message=f"Message delivery timeout after {timeout}s"
        )
    
    def _update_delivery_stats(self, agent_id: str, result: MessageDeliveryResult):
        """Update delivery statistics"""
        with self._lock:
            agent_stats = self.queue_stats[agent_id]
            
            if result.success:
                agent_stats.delivered_messages += 1
                self.global_stats.delivered_messages += 1
                
                # Update average delivery time
                if agent_stats.delivered_messages > 1:
                    agent_stats.average_delivery_time = (
                        (agent_stats.average_delivery_time * (agent_stats.delivered_messages - 1) + 
                         result.delivery_time) / agent_stats.delivered_messages
                    )
                else:
                    agent_stats.average_delivery_time = result.delivery_time
            else:
                agent_stats.failed_messages += 1
                self.global_stats.failed_messages += 1
            
            agent_stats.retry_attempts += result.retry_count
            agent_stats.queue_size = len(self.agent_queues[agent_id])
    
    def get_queue_stats(self, agent_id: str = None) -> Union[MessageQueueStats, Dict[str, MessageQueueStats]]:
        """Get queue statistics"""
        if agent_id:
            return self.queue_stats.get(agent_id, MessageQueueStats())
        else:
            return dict(self.queue_stats)
    
    def get_global_stats(self) -> MessageQueueStats:
        """Get global message broker statistics"""
        return self.global_stats

class AgentRegistry:
    """Registry for tracking agent status and capabilities"""
    
    def __init__(self, heartbeat_interval: float = 30.0, timeout_threshold: float = 90.0):
        self.heartbeat_interval = heartbeat_interval
        self.timeout_threshold = timeout_threshold
        
        # Agent tracking
        self.registered_agents: Dict[str, AgentStatus] = {}
        self.agent_instances: Dict[str, Any] = {}  # weak references to actual agents
        self.agent_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Status monitoring
        self._monitoring_task = None
        self._running = True
        self._lock = threading.Lock()
        
        logger.info("Agent Registry initialized")
    
    async def start(self):
        """Start the agent registry monitoring"""
        if self._monitoring_task is None:
            self._running = True
            self._monitoring_task = asyncio.create_task(self._monitor_agents())
            logger.info("Agent Registry monitoring started")
    
    async def stop(self):
        """Stop the agent registry monitoring"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Agent Registry monitoring stopped")
    
    def register_agent(self, agent_instance, agent_type: str = None) -> bool:
        """Register an agent with the registry"""
        try:
            # Get agent info
            agent_id = getattr(agent_instance, 'agent_id', getattr(agent_instance, 'name', str(uuid.uuid4())))
            
            if agent_type is None:
                if hasattr(agent_instance, 'capabilities'):
                    agent_type = agent_instance.capabilities.agent_type
                else:
                    agent_type = 'unknown'
            
            # Get capabilities
            capabilities = None
            if hasattr(agent_instance, 'capabilities'):
                capabilities = asdict(agent_instance.capabilities)
            
            # Create agent status
            status = AgentStatus(
                agent_id=agent_id,
                agent_type=agent_type,
                capabilities=capabilities
            )
            
            with self._lock:
                self.registered_agents[agent_id] = status
                self.agent_instances[agent_id] = weakref.ref(agent_instance)
            
            logger.info(f"Registered agent: {agent_id} (type: {agent_type})")
            
            # Notify callbacks
            self._notify_agent_callbacks(agent_id, 'registered')
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent: {e}")
            return False
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the registry"""
        try:
            with self._lock:
                if agent_id in self.registered_agents:
                    del self.registered_agents[agent_id]
                if agent_id in self.agent_instances:
                    del self.agent_instances[agent_id]
            
            logger.info(f"Unregistered agent: {agent_id}")
            
            # Notify callbacks
            self._notify_agent_callbacks(agent_id, 'unregistered')
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    def update_agent_heartbeat(self, agent_id: str):
        """Update agent heartbeat timestamp"""
        with self._lock:
            if agent_id in self.registered_agents:
                self.registered_agents[agent_id].last_heartbeat = time.time()
    
    def update_agent_stats(self, agent_id: str, completed_tasks: int = None, failed_tasks: int = None, 
                          active_tasks: int = None, response_time: float = None):
        """Update agent performance statistics"""
        with self._lock:
            if agent_id in self.registered_agents:
                agent = self.registered_agents[agent_id]
                
                if completed_tasks is not None:
                    agent.completed_tasks = completed_tasks
                if failed_tasks is not None:
                    agent.failed_tasks = failed_tasks
                if active_tasks is not None:
                    agent.active_tasks = active_tasks
                if response_time is not None:
                    # Update average response time
                    if agent.completed_tasks > 1:
                        agent.average_response_time = (
                            (agent.average_response_time * (agent.completed_tasks - 1) + response_time) 
                            / agent.completed_tasks
                        )
                    else:
                        agent.average_response_time = response_time
    
    def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """Get status for a specific agent"""
        return self.registered_agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: str) -> List[AgentStatus]:
        """Get all agents of a specific type"""
        return [agent for agent in self.registered_agents.values() 
                if agent.agent_type == agent_type]
    
    def get_available_agents(self, agent_type: str = None) -> List[AgentStatus]:
        """Get all available (active) agents"""
        current_time = time.time()
        available = []
        
        for agent in self.registered_agents.values():
            # Check if agent is still responding
            if current_time - agent.last_heartbeat <= self.timeout_threshold:
                if agent_type is None or agent.agent_type == agent_type:
                    available.append(agent)
        
        return available
    
    def get_agent_instance(self, agent_id: str) -> Optional[Any]:
        """Get the actual agent instance (if still alive)"""
        if agent_id in self.agent_instances:
            weak_ref = self.agent_instances[agent_id]
            return weak_ref() if weak_ref() is not None else None
        return None
    
    def add_agent_callback(self, agent_id: str, callback: Callable):
        """Add callback for agent status changes"""
        self.agent_callbacks[agent_id].append(callback)
    
    def remove_agent_callback(self, agent_id: str, callback: Callable):
        """Remove agent callback"""
        if agent_id in self.agent_callbacks:
            try:
                self.agent_callbacks[agent_id].remove(callback)
            except ValueError:
                pass
    
    async def _monitor_agents(self):
        """Monitor agent health and status"""
        while self._running:
            try:
                current_time = time.time()
                timed_out_agents = []
                
                # Check for timed out agents
                with self._lock:
                    for agent_id, agent in self.registered_agents.items():
                        if current_time - agent.last_heartbeat > self.timeout_threshold:
                            agent.status = "timeout"
                            timed_out_agents.append(agent_id)
                
                # Handle timed out agents
                for agent_id in timed_out_agents:
                    logger.warning(f"Agent {agent_id} timed out")
                    self._notify_agent_callbacks(agent_id, 'timeout')
                
                # Sleep until next check
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in agent monitoring: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    def _notify_agent_callbacks(self, agent_id: str, event: str):
        """Notify callbacks about agent events"""
        for callback in self.agent_callbacks.get(agent_id, []):
            try:
                callback(agent_id, event)
            except Exception as e:
                logger.error(f"Error in agent callback: {e}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        current_time = time.time()
        active_agents = sum(1 for agent in self.registered_agents.values() 
                          if current_time - agent.last_heartbeat <= self.timeout_threshold)
        
        agent_types = defaultdict(int)
        for agent in self.registered_agents.values():
            agent_types[agent.agent_type] += 1
        
        return {
            'total_registered': len(self.registered_agents),
            'active_agents': active_agents,
            'timed_out_agents': len(self.registered_agents) - active_agents,
            'agent_types': dict(agent_types),
            'monitoring_interval': self.heartbeat_interval,
            'timeout_threshold': self.timeout_threshold
        }

class AgentCommunicationSystem:
    """Main communication system that coordinates all components"""
    
    def __init__(self, max_queue_size: int = 10000, max_workers: int = 4,
                 heartbeat_interval: float = 30.0, timeout_threshold: float = 90.0):
        
        self.status = CommunicationStatus.ACTIVE
        
        # Initialize components
        self.message_broker = MessageBroker(max_queue_size, max_workers)
        self.agent_registry = AgentRegistry(heartbeat_interval, timeout_threshold)
        
        # Error handling
        self.error_handlers: List[Callable] = []
        self.performance_monitors: List[Callable] = []
        
        # Statistics
        self.system_stats = {
            'start_time': time.time(),
            'total_messages_processed': 0,
            'errors_handled': 0,
            'uptime': 0.0
        }
        
        logger.info("Agent Communication System initialized")
    
    async def start(self):
        """Start the communication system"""
        try:
            await self.message_broker.start()
            await self.agent_registry.start()
            
            self.status = CommunicationStatus.ACTIVE
            logger.info("Agent Communication System started")
            
        except Exception as e:
            self.status = CommunicationStatus.ERROR
            logger.error(f"Failed to start communication system: {e}")
            raise
    
    async def stop(self):
        """Stop the communication system"""
        try:
            self.status = CommunicationStatus.SHUTDOWN
            
            await self.message_broker.stop()
            await self.agent_registry.stop()
            
            logger.info("Agent Communication System stopped")
            
        except Exception as e:
            logger.error(f"Error stopping communication system: {e}")
    
    def register_agent(self, agent_instance, agent_type: str = None) -> bool:
        """Register an agent with the system"""
        return self.agent_registry.register_agent(agent_instance, agent_type)
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the system"""
        return self.agent_registry.unregister_agent(agent_id)
    
    async def send_message(self, message: AgentMessage, timeout: float = 30.0) -> MessageDeliveryResult:
        """Send a message through the system"""
        if self.status != CommunicationStatus.ACTIVE:
            return MessageDeliveryResult(
                message_id=message.id,
                success=False,
                error_message=f"Communication system not active: {self.status.value}"
            )
        
        try:
            result = await self.message_broker.send_message(message, timeout)
            self.system_stats['total_messages_processed'] += 1
            return result
            
        except Exception as e:
            self._handle_error(f"Message sending failed: {e}")
            return MessageDeliveryResult(
                message_id=message.id,
                success=False,
                error_message=str(e)
            )
    
    async def send_task_to_agent(self, agent_id: str, task: TaskSpecification) -> ProcessingResult:
        """Send a task to a specific agent and await result"""
        # Create task message
        message = AgentMessage(
            sender="communication_system",
            receiver=agent_id,
            message_type=MessageType.TASK,
            payload={
                'task_id': task.task_id,
                'task_type': task.task_type,
                'input_data': task.input_data,
                'requirements': task.requirements,
                'priority': task.priority.value,
                'timeout': task.timeout,
                'retry_count': task.retry_count,
                'metadata': task.metadata
            },
            priority=task.priority
        )
        
        # Send message
        delivery_result = await self.send_message(message)
        
        if delivery_result.success:
            # Parse response into ProcessingResult
            response_data = delivery_result.agent_response or {}
            return ProcessingResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                output_data=response_data,
                processing_time=delivery_result.delivery_time
            )
        else:
            return ProcessingResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error_message=delivery_result.error_message
            )
    
    def get_available_agents(self, agent_type: str = None) -> List[AgentStatus]:
        """Get available agents"""
        return self.agent_registry.get_available_agents(agent_type)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        current_time = time.time()
        self.system_stats['uptime'] = current_time - self.system_stats['start_time']
        
        return {
            'status': self.status.value,
            'system_stats': self.system_stats,
            'message_broker_stats': self.message_broker.get_global_stats(),
            'agent_registry_stats': self.agent_registry.get_registry_stats(),
            'timestamp': current_time
        }
    
    def add_error_handler(self, handler: Callable):
        """Add error handler"""
        self.error_handlers.append(handler)
    
    def add_performance_monitor(self, monitor: Callable):
        """Add performance monitor"""
        self.performance_monitors.append(monitor)
    
    def _handle_error(self, error_message: str):
        """Handle system errors"""
        logger.error(error_message)
        self.system_stats['errors_handled'] += 1
        
        for handler in self.error_handlers:
            try:
                handler(error_message)
            except Exception as e:
                logger.error(f"Error in error handler: {e}")

# Global communication system instance
_communication_system = None

def get_communication_system(**kwargs) -> AgentCommunicationSystem:
    """Get or create the global communication system instance"""
    global _communication_system
    if _communication_system is None:
        _communication_system = AgentCommunicationSystem(**kwargs)
    return _communication_system

async def initialize_communication_system(**kwargs):
    """Initialize and start the global communication system"""
    system = get_communication_system(**kwargs)
    await system.start()
    return system

async def shutdown_communication_system():
    """Shutdown the global communication system"""
    global _communication_system
    if _communication_system:
        await _communication_system.stop()
        _communication_system = None

# Export classes
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