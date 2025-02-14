from typing import Dict, Any, Optional

class StateManager:
    """Manages conversation state and context."""
    
    def __init__(self):
        self.state = {
            "travel_dates": None,
            "season_info": {},
            "interests": [],
            "itinerary": [],
            "current_step": "greeting",
            "last_input": None,
            "current_topic": None,
            "last_suggestions": []
        }
        self.conversation_history = []
    
    def update_state(self, key: str, value: Any) -> None:
        """Update a state value."""
        self.state[key] = value
    
    def get_state(self, key: str) -> Optional[Any]:
        """Get a state value."""
        return self.state.get(key)
    
    def add_to_conversation(self, user_input: str, bot_response: str) -> None:
        """Add an exchange to conversation history."""
        self.conversation_history.append({
            "user": user_input,
            "bot": bot_response
        })
    
    def get_context(self) -> Dict[str, Any]:
        """Get current context for LLM."""
        return {
            "current_interests": ", ".join(self.state["interests"]),
            "current_topic": self.state.get("current_topic", "None"),
            "places_in_itinerary": len(self.state["itinerary"]),
            "travel_dates": self.state.get("travel_dates", "unknown"),
            "season_info": self.state.get("season_info", {}),
            "conversation_stage": self.state["current_step"]
        } 