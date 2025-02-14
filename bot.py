from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import StateManager
from router import IntentRouter
from handlers import (
    ItineraryHandler, SearchHandler, 
    QuestionHandler, DateHandler,
    ThankYouHandler
)
from prompts import QUERY_ANALYSIS_PROMPT

# Date validation prompt
DATE_VALIDATION_PROMPT = PromptTemplate(
    input_variables=["date_input", "current_date"],
    template="""As a Puerto Rico travel expert, analyze this user message and extract/validate their travel date: "{date_input}"
    
    Current date: {current_date}
    
    First, extract any date-related information from the user's message. Look for:
    - Explicit dates ("December 2025")
    - Relative dates ("next summer", "in 3 months")
    - Casual mentions ("visiting in December", "planning a trip next year")
    
    Then validate the extracted date considering these seasons in Puerto Rico:
    1. High Season (Mid-December to Mid-April):
       - Peak tourist season
       - Dry season with less rainfall
       - Average temperatures 75-85°F
       - Perfect for beach activities and outdoor exploration
       - Higher prices and more crowded
    
    2. Shoulder Season (April to June):
       - Moderate tourist traffic
       - Increasing humidity
       - Average temperatures 80-90°F
       - Good for rainforest visits and water activities
       - Better prices on accommodations
    
    3. Low Season (July to November):
       - Hurricane season
       - Frequent afternoon showers
       - Average temperatures 85-95°F
       - Best prices and fewer tourists
       - Need to monitor weather forecasts
       - Many indoor alternatives available
    
    Rules for validation:
    1. Date must be after {current_date}
    2. Format any date input into a clear date
    3. Consider hurricane season risks (July-November)
    
    Format your response exactly like this:
    VALID: [formatted_date] | [season_type] | [weather_expectations] | [travel_tips]
    or 
    INVALID: [reason] | [suggestion]

    Example responses:
    VALID: December 15, 2024 | High Season | Dry season with less rainfall, average temperatures 75-85°F | Book accommodations early as this is peak tourist season
    INVALID: Unable to determine a specific date from the message | Please provide a clearer date like "December 2024" or "next summer"
    """
)

# Interest analysis prompt
INTEREST_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["interests"],
    template="""Analyze these travel interests for Puerto Rico: {interests}

    You must respond in this exact format with the | symbol:
    CATEGORIES: [comma-separated list of matching categories] | KEYWORDS: [key search terms]

    Available categories:
    1. Nature/Outdoor (beaches, hiking, rainforest, etc.)
    2. Cultural/Historical (museums, architecture, etc.)
    3. Adventure/Sports (surfing, diving, etc.)
    4. Food/Dining (restaurants, food tours, etc.)
    5. Entertainment (nightlife, shows, etc.)

    Example response:
    CATEGORIES: Nature/Outdoor, Adventure/Sports | KEYWORDS: beaches, surfing, water activities

    Remember to always include the | symbol between CATEGORIES and KEYWORDS."""
)

# Query analysis prompt
QUERY_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["user_input", "current_context"],
    template="""Analyze this user's travel query for Puerto Rico: "{user_input}"

    Current conversation context: {current_context}

    Extract the following information in this exact format with | symbols:
    INTENT: [search_places|show_interest|add_to_itinerary|show_itinerary|ask_question|finalize|thanking|other]
    SEARCH_TYPE: [attractions|restaurants|beaches|museums|churches|activities|etc]
    LOCATION: [specific area or 'any']
    SPECIFICS: [type=historical, cuisine=local, activity=hiking, selections=1,2, etc]
    QUERY: [reformulated search query]

    Intent Categories:
    - add_to_itinerary: Any request to add items to list, including:
      * Number-based selections ("add 1 and 2", "add 1,2,3,4", "add 1,2 and 3")
      * Multiple selections ("add first and second")
      * All items ("add all", "add everything")
    - search_places: Looking for specific places or recommendations
    - show_interest: Expressing interest in certain types of places/activities
    - show_itinerary: Want to see their current list
    - ask_question: Asking for specific information
    - finalize: Want to complete their itinerary
    - thanking: Any expression of gratitude ("thank you", "thanks", "gracias")

    Examples:
    Input: "add 1 and 3 to my list"
    INTENT: add_to_itinerary | SEARCH_TYPE: any | LOCATION: any | SPECIFICS: selections=1,3 | QUERY: Add items 1 and 3 from the last suggestions

    Input: "add numbers 2 and 4"
    INTENT: add_to_itinerary | SEARCH_TYPE: any | LOCATION: any | SPECIFICS: selections=2,4 | QUERY: Add items 2 and 4 from suggestions

    Input: "add all of them"
    INTENT: add_to_itinerary | SEARCH_TYPE: any | LOCATION: any | SPECIFICS: selections=all | QUERY: Add all suggested items

    Remember to:
    1. Prioritize identifying add/save commands when numbers are mentioned
    2. Look for patterns like "add X and Y", "add X,Y,Z"
    3. Consider context of previous suggestions
    4. Handle both formal and informal language
    """
)

class SimplePRTravelBot:
    """Main bot class using NLP-driven architecture."""

    def __init__(self, llm, retriever, index, location_chain):
        """Initialize bot with core components."""
        # Initialize state manager
        self.state_manager = StateManager()
        
        # Initialize handlers
        handlers = {
            "date": DateHandler(self.state_manager, llm),
            "search": SearchHandler(retriever, index, llm, location_chain, self.state_manager),
            "question": QuestionHandler(retriever, llm, self.state_manager),
            "itinerary": ItineraryHandler(self.state_manager),
            "thankyou": ThankYouHandler()
        }
        
        # Initialize router
        self.router = IntentRouter(handlers)
        
        # Initialize LLM components
        self.llm = llm
        self.query_chain = QUERY_ANALYSIS_PROMPT | llm | StrOutputParser()

    async def _process_input(self, user_input: str) -> str:
        """Process user input using NLP-driven routing."""
        try:
            # Get current context
            context = self.state_manager.get_context()
            
            # Quick exit check
            if user_input.lower() in ['exit', 'quit', 'bye', 'thanks. Bye', 'goodbye', 'lets finish', 'thats all', 'that is all', 'thanks bye', 'thank you bye', 'thanks, that will be all', 'thanks, that will be all. bye']:
                itinerary = self.state_manager.get_state("itinerary")
                travel_dates = self.state_manager.get_state("travel_dates")
                
                # Format itinerary items
                formatted_items = []
                for item in itinerary:
                    display_name = item.replace('_', ' ').title()
                    formatted_items.append(f"✅ {display_name}")

                 # Get season information
                season = season_info.get("season", "")
                weather = season_info.get("weather", "")
                tips = season_info.get("tips", "")
                
                return f"""
                🎉 Thanks for planning your trip to Puerto Rico!
                Here's your final list of places to visit for {travel_dates}:

                {chr(10).join(formatted_items)}

                🌡️ Season Information:
                {season}

                🌤️ Weather Expectations:
                {weather}

                💡 Travel Tips:
                {tips}

                Additional Resources:
                • 🎯 For events, restaurants and deals visit: https://app.voyturisteando.com/ & https://www.discoverpuertorico.com/
                • 🌤️ For accurate weather information visit: https://www.weather.gov/sju/ & https://www.caricoos.org/

                👋 Have a great journey! ¡Buen viaje! 🌴 
                Feel free to come back anytime, if you have any questions or need help planning your next trip.
                """
            
            # Add user input to context
            context['user_input'] = user_input
            
            # Check if we're waiting for a date
            if not self.state_manager.get_state("travel_dates"):
                # Try to handle as date input first
                try:
                    date_handler = self.router.handlers.get('date')
                    if date_handler:
                        return await date_handler.handle(context)
                except Exception as e:
                    print(f"Date handling failed: {str(e)}")
            
            # If not a date or date handling failed, proceed with normal intent analysis
            analysis = await self.query_chain.ainvoke({
                "user_input": user_input,
                "current_context": context
            })
            
            # Parse analysis
            parts = analysis.split("|")
            if len(parts) < 5:  # Handle incomplete analysis
                return "Sorry, I'm having trouble understanding. Could you please rephrase that?"
                
            intent = parts[0].replace("INTENT:", "").strip()
            
            # Update context with analysis
            context.update({
                "intent": intent,
                "search_type": parts[1].replace("SEARCH_TYPE:", "").strip(),
                "location": parts[2].replace("LOCATION:", "").strip(),
                "specifics": parts[3].replace("SPECIFICS:", "").strip(),
                "query": parts[4].replace("QUERY:", "").strip()
            })
            
            # Route to appropriate handler
            response = await self.router.route(intent, context)
            
            # Store conversation
            self.state_manager.add_to_conversation(user_input, response)
            
            return response

        except Exception as e:
            print(f"Error in _process_input: {str(e)}")
            return "Sorry, I encountered an error. Could you rephrase that?"

    async def start_chat(self):
        """Start the conversation."""
        welcome = """
        ¡Hola! 😊 I'm your Puerto Rico Travel Assistant.
        
        I'll be helping you plan your trip to Puerto Rico. Let's get started!
        When are you planning to visit our beautiful island🏝️?
        """
        
        print("\nBot:", welcome)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                
                response = await self._process_input(user_input)
                print("\nBot:", response)
                
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Let's try again...")