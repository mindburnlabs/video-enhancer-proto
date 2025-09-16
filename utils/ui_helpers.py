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

import gradio as gr
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import traceback

from utils.error_handler import (
    VideoEnhancementError, ErrorCode, error_handler
)

logger = logging.getLogger(__name__)

class UIFeedbackManager:
    """Manages user feedback, error messages, and progress indicators in Gradio UI"""
    
    def __init__(self):
        self.progress_messages = {
            "uploading": "📤 Uploading your video...",
            "analyzing": "🔍 Analyzing video quality and content...",
            "planning": "📋 Planning enhancement strategy...",
            "enhancing": "✨ Enhancing your video with AI models...",
            "postprocessing": "🎬 Finalizing enhanced video...",
            "completing": "✅ Processing complete!"
        }
        
        self.error_emojis = {
            ErrorCode.INPUT_FILE_NOT_FOUND: "📁",
            ErrorCode.INPUT_INVALID_FORMAT: "🎥",
            ErrorCode.INPUT_FILE_TOO_LARGE: "📏",
            ErrorCode.INPUT_FILE_CORRUPTED: "💔",
            ErrorCode.MODEL_LOAD_ERROR: "🤖",
            ErrorCode.MODEL_MEMORY_ERROR: "🧠",
            ErrorCode.SYSTEM_MEMORY_ERROR: "💾",
            ErrorCode.PROC_ENHANCEMENT_FAILED: "⚠️",
        }
        
    def create_error_message(
        self, 
        error: Union[Exception, VideoEnhancementError],
        user_friendly: bool = True
    ) -> str:
        """Create a user-friendly error message for display in Gradio"""
        
        if isinstance(error, VideoEnhancementError):
            emoji = self.error_emojis.get(error.error_code, "❌")
            
            if user_friendly:
                message = f"{emoji} {error.context.user_message}"
                
                # Add suggestions if available
                if error.context.suggestions:
                    message += "\n\n💡 **Suggestions:**\n"
                    for i, suggestion in enumerate(error.context.suggestions[:3], 1):
                        message += f"{i}. {suggestion}\n"
                
                # Add retry information
                if error.context.retry_possible:
                    message += "\n🔄 You can try again with different settings."
                
                if error.context.fallback_available:
                    message += "\n🛟 Alternative processing methods are available."
                    
            else:
                # Technical message for debugging
                message = f"{emoji} **Technical Error:** {error.message}\n"
                message += f"**Error Code:** {error.error_code.value}\n"
                message += f"**Component:** {error.context.component}\n"
                message += f"**Operation:** {error.context.operation}"
                
        else:
            # Handle regular exceptions
            emoji = "❌"
            if user_friendly:
                message = f"{emoji} An unexpected error occurred: {str(error)}"
                message += "\n\n💡 **Suggestions:**\n"
                message += "1. Please try again\n"
                message += "2. Check your input file\n" 
                message += "3. Contact support if the problem persists"
            else:
                message = f"{emoji} **Exception:** {str(error)}\n"
                message += f"**Type:** {type(error).__name__}"
        
        return message
    
    def create_progress_html(
        self, 
        current_stage: str,
        progress_percent: float = 0,
        stages_completed: List[str] = None,
        stages_remaining: List[str] = None,
        estimated_time_remaining: Optional[int] = None
    ) -> str:
        """Create HTML progress indicator for Gradio"""
        
        stages_completed = stages_completed or []
        stages_remaining = stages_remaining or []
        
        # Progress bar HTML
        progress_html = f"""
        <div style="margin: 20px 0;">
            <div style="background-color: #f0f0f0; border-radius: 10px; padding: 3px;">
                <div style="background-color: #4CAF50; width: {progress_percent}%; height: 30px; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                    {progress_percent:.1f}%
                </div>
            </div>
        """
        
        # Current stage
        stage_message = self.progress_messages.get(current_stage, f"Processing: {current_stage}")
        progress_html += f"""
            <div style="margin: 10px 0; font-size: 16px; font-weight: bold; color: #333;">
                {stage_message}
            </div>
        """
        
        # Time remaining
        if estimated_time_remaining:
            mins, secs = divmod(estimated_time_remaining, 60)
            if mins > 0:
                time_str = f"{mins}m {secs}s"
            else:
                time_str = f"{secs}s"
            
            progress_html += f"""
                <div style="margin: 5px 0; color: #666; font-size: 14px;">
                    ⏱️ Estimated time remaining: {time_str}
                </div>
            """
        
        # Stages overview
        if stages_completed or stages_remaining:
            progress_html += """
                <div style="margin: 15px 0; padding: 10px; background-color: #f9f9f9; border-radius: 5px;">
                    <div style="font-weight: bold; margin-bottom: 8px;">Processing Pipeline:</div>
            """
            
            # Completed stages
            for stage in stages_completed:
                progress_html += f"""
                    <div style="margin: 3px 0; color: #4CAF50;">
                        ✅ {stage.replace('_', ' ').title()}
                    </div>
                """
            
            # Current stage
            if current_stage:
                progress_html += f"""
                    <div style="margin: 3px 0; color: #2196F3; font-weight: bold;">
                        🔄 {current_stage.replace('_', ' ').title()}
                    </div>
                """
            
            # Remaining stages
            for stage in stages_remaining:
                progress_html += f"""
                    <div style="margin: 3px 0; color: #999;">
                        ⏳ {stage.replace('_', ' ').title()}
                    </div>
                """
            
            progress_html += "</div>"
        
        progress_html += "</div>"
        return progress_html
    
    def create_success_message(
        self,
        processing_time: float,
        model_used: str,
        quality_improvement: Optional[float] = None,
        file_size_mb: Optional[float] = None
    ) -> str:
        """Create success message with processing statistics"""
        
        mins, secs = divmod(int(processing_time), 60)
        time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
        
        message = f"""
        🎉 **Video Enhancement Complete!**
        
        ✅ **Processing Time:** {time_str}
        🤖 **Model Used:** {model_used}
        """
        
        if quality_improvement:
            message += f"\n📈 **Quality Improvement:** {quality_improvement:.1%}"
        
        if file_size_mb:
            message += f"\n📁 **Output Size:** {file_size_mb:.1f} MB"
        
        message += """
        
        💡 **Tips:**
        • You can download your enhanced video below
        • Try different settings for various results  
        • Share your feedback to help us improve!
        """
        
        return message
    
    def create_info_panel(
        self,
        title: str,
        content: Dict[str, Any],
        panel_type: str = "info"  # info, warning, error, success
    ) -> str:
        """Create an informational panel for Gradio"""
        
        colors = {
            "info": {"bg": "#e3f2fd", "border": "#2196f3", "icon": "ℹ️"},
            "warning": {"bg": "#fff3e0", "border": "#ff9800", "icon": "⚠️"},
            "error": {"bg": "#ffebee", "border": "#f44336", "icon": "❌"},
            "success": {"bg": "#e8f5e8", "border": "#4caf50", "icon": "✅"}
        }
        
        color_scheme = colors.get(panel_type, colors["info"])
        
        html = f"""
        <div style="
            margin: 15px 0; 
            padding: 15px; 
            background-color: {color_scheme['bg']}; 
            border-left: 4px solid {color_scheme['border']}; 
            border-radius: 5px;
        ">
            <div style="font-weight: bold; margin-bottom: 10px; color: #333;">
                {color_scheme['icon']} {title}
            </div>
        """
        
        for key, value in content.items():
            html += f"""
                <div style="margin: 5px 0; color: #555;">
                    <strong>{key.replace('_', ' ').title()}:</strong> {value}
                </div>
            """
        
        html += "</div>"
        return html

def create_enhanced_error_display() -> gr.HTML:
    """Create enhanced error display component for Gradio"""
    return gr.HTML(
        value="",
        visible=False,
        label="Error Information"
    )

def create_progress_display() -> gr.HTML:
    """Create progress display component for Gradio"""
    return gr.HTML(
        value="",
        visible=False,
        label="Processing Progress"
    )

def create_success_display() -> gr.HTML:
    """Create success message display component for Gradio"""
    return gr.HTML(
        value="",
        visible=False,
        label="Success Information"
    )

def handle_gradio_error(
    error: Exception,
    feedback_manager: UIFeedbackManager,
    show_technical: bool = False
) -> Tuple[str, bool]:
    """Handle errors in Gradio interface and return formatted message"""
    
    try:
        # Convert to VideoEnhancementError if needed
        if not isinstance(error, VideoEnhancementError):
            enhanced_error = error_handler.handle_error(
                error=error,
                component="ui",
                operation="user_interaction",
                user_message="An error occurred while processing your request",
                suggestions=[
                    "Please try again with different settings",
                    "Check that your video file is valid and not corrupted",
                    "Contact support if the problem persists"
                ]
            )
        else:
            enhanced_error = error
        
        # Create user-friendly error message
        error_message = feedback_manager.create_error_message(
            enhanced_error, 
            user_friendly=not show_technical
        )
        
        return error_message, True  # (message, visible)
        
    except Exception as e:
        logger.error(f"Error in error handling: {e}")
        return f"❌ An unexpected error occurred: {str(error)}", True

def update_progress_display(
    current_stage: str,
    progress_percent: float,
    feedback_manager: UIFeedbackManager,
    **kwargs
) -> Tuple[str, bool]:
    """Update progress display in Gradio"""
    
    try:
        progress_html = feedback_manager.create_progress_html(
            current_stage=current_stage,
            progress_percent=progress_percent,
            **kwargs
        )
        
        return progress_html, True  # (html, visible)
        
    except Exception as e:
        logger.error(f"Error updating progress: {e}")
        return f"Processing: {current_stage} ({progress_percent:.1f}%)", True

def clear_displays() -> Tuple[Tuple[str, bool], Tuple[str, bool], Tuple[str, bool]]:
    """Clear all display components"""
    return (
        ("", False),  # error_display
        ("", False),  # progress_display
        ("", False)   # success_display
    )

# Global UI feedback manager instance
ui_feedback = UIFeedbackManager()