from langchain.prompts import PromptTemplate

QUERY_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["user_input", "current_context"],
    template="""Analyze this user's travel query for Puerto Rico: "{user_input}"

    Current conversation context: {current_context}

    First, check if this is a date-related input by looking for:
    - Explicit dates ("December 2024", "12/2024")
    - Relative dates ("in 3 months", "next summer")
    - Casual date mentions ("visiting in December", "planning for next year")
    - Time expressions ("three months from now", "next month")

    If the input contains ANY date information → classify as:
    INTENT: set_date | SEARCH_TYPE: any | LOCATION: any | SPECIFICS: date_input | QUERY: Process travel date

    If the input contains ANY gratitude expressions, check for:
    - Direct thanks ("thank you", "thanks", "thank you so much")
    - Informal thanks ("thx", "ty")
    - Spanish thanks ("gracias", "muchas gracias")
    - Appreciation ("appreciate it", "that was helpful")
    → classify as:
    INTENT: thanking | SEARCH_TYPE: any | LOCATION: any | SPECIFICS: gratitude | QUERY: Process thank you

    Only if NO date information is found, then classify as one of:
    • Asking for information ("tell me about X", "what is Y")
    • Expressing interest ("show me beaches", "find museums")
    • Managing itinerary ("add this", "show list")
    • Ending conversation ("that's all", "let's finish")

    Format response exactly with | symbols:
    INTENT: [set_date|qa_about_place|discover_places|add_to_itinerary|show_itinerary|finalize|thanking|other]
    SEARCH_TYPE: [specific_place|attractions|restaurants|beaches|museums|activities|etc]
    LOCATION: [specific area or 'any']
    SPECIFICS: [any relevant details about the request]
    QUERY: [natural language reformulation of the request]
    """
)

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