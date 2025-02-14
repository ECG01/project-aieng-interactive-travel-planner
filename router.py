from typing import Dict
from handlers import BaseHandler

class IntentRouter:
    """Routes intents to appropriate handlers."""

    def __init__(self, handlers: Dict[str, BaseHandler]):
        """Initialize with a map of intent types to handlers."""
        self.handlers = handlers

    async def route(self, intent: str, context: Dict) -> str:
        """Route to appropriate handler based on intent."""
        try:
            # Map intents to handlers
            intent_map = {
                'set_date': 'date',
                'qa_about_place': 'question',
                'discover_places': 'search',
                'search_places': 'search',
                'add_to_itinerary': 'itinerary',
                'show_itinerary': 'itinerary',
                'finalize': 'itinerary',
                'thanking': 'thankyou'
            }
            
            # Get handler type
            handler_type = intent_map.get(intent)
            
            if not handler_type:
                # Check if input might be a date
                if any(word in context['user_input'].lower() for word in 
                      ['month', 'year', 'summer', 'winter', 'spring', 'fall', 'next', 
                       'january', 'february', 'march', 'april', 'may', 'june', 'july',
                       'august', 'september', 'october', 'november', 'december']):
                    handler_type = 'date'
                else:
                    return "I'm not sure what you'd like to do. Could you rephrase that?"
            
            # Get handler
            handler = self.handlers.get(handler_type)
            if not handler:
                return "I'm not sure how to handle that request."
            
            # Execute handler
            return await handler.handle(context)
            
        except Exception as e:
            print(f"Error in router: {str(e)}")
            return "I encountered an error. Could you rephrase that?" 