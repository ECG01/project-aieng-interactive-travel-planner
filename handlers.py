from typing import Dict, Any, List, Tuple
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import re
from chains.qa_chain import PlaceQAChain
from state import StateManager
from prompts import DATE_VALIDATION_PROMPT
from langchain_core.output_parsers import StrOutputParser
import dateparser
from difflib import get_close_matches

class BaseHandler(ABC):
    """Base class for all handlers."""
    
    @abstractmethod
    async def handle(self, context: Dict[str, Any]) -> str:
        """Handle the intent with given context."""
        pass

class ItineraryHandler(BaseHandler):
    """Handler for itinerary-related intents."""
    
    def __init__(self, state_manager):
        self.state = state_manager
    
    async def handle(self, context: Dict[str, Any]) -> str:
        intent = context.get("intent")
        if intent == "add_to_itinerary":
            return await self._add_items(context)
        elif intent == "show_itinerary":
            return self._display_itinerary()
        elif intent == "finalize":

            # Get season information
            season_info = self.state.get_state("season_info")
            season = season_info.get("season", "")
            weather = season_info.get("weather", "")
            tips = season_info.get("tips", "")

            # Show final itinerary and goodbye message
            return f"""
            🎉 Perfect! Here's your final list of places and activities for {self.state.get_state('travel_dates')}:

            {self._format_itinerary()}

            🌡️ Season Information:
            {season}

            🌤️ Weather Expectations:
            {weather}

            💡 Travel Tips:
            {tips}

            Additional Resources:
                • 🎯 For events, restaurants and deals visit: https://app.voyturisteando.com/ & https://www.discoverpuertorico.com/ 
                • 🌤️ For accurate weather information visit: https://www.weather.gov/sju/ & https://www.caricoos.org

            👋 Have a great journey! ¡Buen viaje! 🌴 
            Feel free to come back anytime, if you have any questions or need help planning your next trip. 
            """
        
        return "Sorry, I'm not sure what you'd like to do with your list."
    
    async def _add_items(self, context: Dict[str, Any]) -> str:
        """Add items to itinerary based on context."""
        try:
            specifics = context.get("specifics", "")
            last_suggestions = self.state.get_state("last_suggestions")
            
            if not last_suggestions:
                return """
                Sorry, I don't have any recent suggestions to add. Let me help you find some places.
                What interests you about Puerto Rico?
                """

            added_places = []
            
            # Handle "add all" case
            if "selections=all" in specifics:
                for suggestion in last_suggestions:
                    name = suggestion.get('name', '')
                    if name and name not in self.state.get_state("itinerary"):
                        self.state.get_state("itinerary").append(name)
                        added_places.append(name)
            else:
                # Handle specific selections
                selections = []
                if "selections=" in specifics:
                    nums = specifics.split("selections=")[1].split(",")
                    selections = [int(n) for n in nums if n.isdigit()]
                
                for num in selections:
                    if 0 < num <= len(last_suggestions):
                        suggestion = last_suggestions[num-1]
                        name = suggestion.get('name', '')
                        if name and name not in self.state.get_state("itinerary"):
                            self.state.get_state("itinerary").append(name)
                            added_places.append(name)

            if added_places:
                formatted_places = [place.replace('_', ' ').title() for place in added_places]
                return f"""
                ✅ Added to your list:
                {chr(10).join(f'• {place}' for place in formatted_places)}

                What else would you like to do?
                • Add other places from the previous search 🔄
                • Tell me about other places you're interested in visiting 🌴
                • View your complete list 📋
                """

            return """
            Please specify which places you'd like to add:
            • Say 'add all' to add all places
            • Specify numbers (e.g., 'add 1 and 3' or 'add 1,2,4')
            """

        except Exception as e:
            print(f"Error in _add_items: {str(e)}")
            return "Sorry, I had trouble updating your list. Please try again."
    
    def _display_itinerary(self) -> str:
        """Display current itinerary."""
        itinerary = self.state.get_state("itinerary")
        if not itinerary:
            return """
            Your list is empty! 
            
            Would you like to:
            • Search for places to visit? 🔍
            • Get suggestions based on your interests? 💡
            • Start over with a new plan? 🆕
            """
        
        formatted_items = []
        for item in itinerary:
            display_name = item.replace('_', ' ').title()
            formatted_items.append(f"✅ {display_name}")
        
        return f"""
        📋 Here's your current list:

        {chr(10).join(formatted_items)}

        Would you like to:
        • Tell me about other places you're interested in visiting 🌴
        • Get details about any place? ℹ️
        • Close the list and finalize your plan? 🔒
        """
    
    def _format_itinerary(self) -> str:
        """Format the itinerary for display."""
        itinerary = self.state.get_state("itinerary")
        if not itinerary:
            return "Your list is empty."
        
        formatted_items = []
        for item in itinerary:
            display_name = item.replace('_', ' ').title()
            formatted_items.append(f"✅ {display_name}")
        
        return chr(10).join(formatted_items)

class SearchHandler(BaseHandler):
    """Handler for search-related intents."""
    
    def __init__(self, retriever, index, llm, location_chain, state_manager):
        self.retriever = retriever
        self.index = index
        self.llm = llm
        self.location_chain = location_chain
        self.state = state_manager
    
    async def handle(self, context: Dict[str, Any]) -> str:
        """Handle search queries."""
        try:
            search_type = context.get("search_type", "any")
            location = context.get("location", "any")
            specifics = context.get("specifics", "")
            query = context.get("query", "")
            
            # Get search results
            results = await self._handle_search(query, search_type, location, specifics)
            
            # Store suggestions in state
            self._store_suggestions(results)
            
            return results
            
        except Exception as e:
            print(f"Error in SearchHandler: {str(e)}")
            return "Sorry, I had trouble searching. Could you try rephrasing your request?"
    
    def _store_suggestions(self, results: str) -> None:
        """Store formatted results as suggestions."""
        suggestions = []
        
        # Parse the formatted results string
        sections = results.split("\n\n")
        for section in sections:
            if section.strip().startswith(("1️⃣", "2️⃣", "3️⃣", "4️⃣")):
                lines = section.strip().split("\n")
                name = lines[0].split("**")[1].strip()
                suggestion = {
                    'name': name,
                    'description': "\n".join(lines[4:]),  # Get description
                    'metadata': {
                        'type': lines[1].replace("🏷️", "").strip(),
                        'location': lines[2].replace("📍", "").strip()
                    }
                }
                suggestions.append(suggestion)
        
        # Store in state
        if self.state:
            self.state.update_state("last_suggestions", suggestions)
    
    async def _handle_search(self, query: str, search_type: str, location: str, specifics: str) -> str:
        """Handle search queries with location chain."""
        try:
            # Build search query
            base_query = self._build_search_query(search_type, location, specifics)
            
            # Use retriever directly for search
            docs = await self.retriever.ainvoke(base_query)
            
            if not docs or not isinstance(docs, list):
                return await self._handle_no_results(search_type, location)
            
            # Convert documents to our format
            formatted_docs = []
            for doc in docs:
                if hasattr(doc, 'page_content'):  # Handle LangChain document format
                    metadata = doc.metadata or {}
                    formatted_docs.append({
                        'name': metadata.get('name', 'Unknown Location'),
                        'content': doc.page_content,
                        'metadata': {
                            'type': metadata.get('type', search_type),
                            'location': metadata.get('location', location),
                            'coordinates': metadata.get('coordinates', 'Coordinates not available'),
                            'town': metadata.get('location', location)
                        }
                    })
                elif isinstance(doc, dict):  # Handle dictionary format
                    formatted_docs.append({
                        'name': doc.get('name', 'Unknown Location'),
                        'content': doc.get('content', doc.get('page_content', '')),
                        'metadata': {
                            'type': doc.get('type', search_type),
                            'location': doc.get('location', location),
                            'coordinates': doc.get('coordinates', 'Coordinates not available'),
                            'town': doc.get('location', location)
                        }
                    })
            
            if not formatted_docs:
                return await self._handle_no_results(search_type, location)
                
            return self._format_search_results(formatted_docs)
            
        except Exception as e:
            print(f"Error in SearchHandler: {str(e)}")
            return "Sorry, I had trouble searching. Could you pleasetry rephrasing your request?"
    
    def _build_search_query(self, search_type: str, location: str, specifics: str) -> str:
        """Build a search query based on type and specifics."""
        location_query = f"in {location}, Puerto Rico" if location != 'any' else "in Puerto Rico"
        
        type_queries = {
            "beaches": f"Find beaches {location_query}. Include popular beaches and activities.",
            "museums": f"Find museums and cultural sites {location_query}.",
            "restaurants": f"Find restaurants and dining options {location_query}.",
            "churches": f"Find churches and religious sites {location_query}.",
            "attractions": f"Find tourist attractions {location_query}."
        }
        
        base_query = type_queries.get(
            search_type.lower(),
            f"Find {search_type} {location_query}"
        )
        
        if specifics:
            base_query += f" Specifically looking for: {specifics}"
            
        return base_query
    
    def _format_search_results(self, docs: List[Dict]) -> str:
        """Format search results with enhanced location info."""
        # Add introduction message
        formatted_results = ["""
        🌟 Based on your interests, here are my suggestions for you:
        """]
        
        for i, doc in enumerate(docs, 1):
            # Get location details
            location = doc.get('metadata', {}).get('location', '')
            coordinates = doc.get('metadata', {}).get('coordinates', 'Coordinates not available')
            town = doc.get('metadata', {}).get('town', 'Town not specified')
            
            # Format the result with number emoji
            number_emoji = f"{i}️⃣"
            name = doc.get('name', '').replace('_', ' ')
            
            result = f"""
                    {number_emoji} **{name}**
                    🏷️ {doc.get('metadata', {}).get('type', 'landmark')}
                    📍 {town}
                    🌐 {coordinates}
                    🔗 [View Images]()
                    💡{doc.get('content', 'No description available')}\n
            """
            formatted_results.append(result)
        
        # Add closing message
        formatted_results.append("""
        Would you like me to add any of these suggestions to your list📝? 
        You can say:
        • "Add all of them"
        • "Add number 1 and 3"
        • "Add 1,2 and 4"
        """)
        
        return "\n\n".join(formatted_results)
    
    async def _handle_no_results(self, search_type: str, location: str) -> str:
        """Handle case when no results are found."""
        nearby_suggestions = ""
        if location != 'any':
            nearby_docs = await self.retriever.ainvoke(
                f"Find {search_type} near {location}, Puerto Rico"
            )
            if nearby_docs:
                locations = set(d.metadata.get('town', '') 
                              for d in nearby_docs[:3] 
                              if hasattr(d, 'metadata'))
                if locations:
                    nearby_suggestions = "\n• Try nearby areas: " + ", ".join(locations)
        
        return f"""
        Sorry, I couldn't find exact matches for {search_type} in {location if location != 'any' else 'Puerto Rico'}.
        
        Would you like to:
        • Broaden your search?{nearby_suggestions}
        • Try different requirements?
        • Look in a different area?
        """
    
    def _extract_coordinates(self, text: str) -> str:
        """Extract coordinates from text or return default."""
        coord_pattern = r'(\d+\.?\d*°?\s*[NS])[\s,]*(\d+\.?\d*°?\s*[EW])'
        match = re.search(coord_pattern, text)
        return match.group(0) if match else 'Coordinates not available'
        
    def _extract_town(self, text: str) -> str:
        """Extract town name from text."""
        # Look for common patterns indicating location
        patterns = [
            r'in\s+([A-Z][a-zA-Z\s]+),\s*Puerto Rico',
            r'located\s+in\s+([A-Z][a-zA-Z\s]+)',
            r'town\s+of\s+([A-Z][a-zA-Z\s]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None

class QuestionHandler(BaseHandler):
    """Handler for question-related intents."""
    
    def __init__(self, retriever, llm, state_manager):
        self.retriever = retriever
        self.qa_chain = PlaceQAChain(llm)
        self.llm = llm
        self.state = state_manager
    
    async def handle(self, context: Dict[str, Any]) -> str:
        """Handle the intent with given context."""
        question = context.get("query", "")
        return await self._handle_question(question)
    
    async def _check_semantic_relevance(self, question: str, docs: List) -> bool:
        """Enhanced semantic relevance check."""
        if not docs or len(docs) == 0:
            return False
            
        # Extract the main subject from the question
        subject = question.lower().replace("tell me about ", "").replace("what is ", "")
        
        # Check if the subject appears in the content
        if subject not in docs[0].page_content.lower():
            return False
            
        # If subject found, do detailed relevance check
        system_prompt = """Evaluate if this content directly answers the question about {subject}.
        Return ONLY 'yes' or 'no'."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\nContent: {docs[0].page_content}"}
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            return response.lower().strip() == 'yes'
        except Exception as e:
            print(f"Error in relevance check: {str(e)}")
            return False
    
    async def _handle_question(self, question: str) -> str:
        """Enhanced question handling with seamless fallback."""
        try:
            # First try vector search
            docs = await self.retriever.ainvoke(question)
            
            # Check semantic relevance
            is_relevant = await self._check_semantic_relevance(question, docs)
            
            if docs and is_relevant:
                # Use vector search results
                doc = docs[0]
                response = await self.qa_chain.ainvoke({
                    "question": question,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "travel_dates": self.state.get_state("travel_dates")
                })
            else:
                # GPT fallback
                response = await self._get_gpt_response(question)
            
            return f"""
            {response}
            
            Would you like to:
            1. Add this place to your list
            2. Ask another question
            3. See more suggestions
            4. Tell me about other interests
            """
            
        except Exception as e:
            print(f"Error in QuestionHandler: {str(e)}")
            return "Sorry, I had trouble answering that. Could you rephrase your question?"
    
    async def _get_gpt_response(self, question: str) -> str:
        """Generate response using GPT."""
        system_prompt = """You are a Puerto Rico Travel Assistant.
        When providing information:
        1. Start by mentioning this is based on general knowledge
        2. Provide accurate historical and cultural context
        3. Include practical visitor information
        4. Suggest official sources for current details
        5. Recommend similar places we might have information about
        6. Format with emojis and clear sections"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Tell me about {question} in Puerto Rico"}
        ]
        
        return await self.llm.ainvoke(messages)

class DateHandler(BaseHandler):
    """Handler for date-related interactions."""
    
    def __init__(self, state_manager, llm):
        self.state = state_manager
        self.date_chain = DATE_VALIDATION_PROMPT | llm | StrOutputParser()
        
        # Common month spellings and variations (English and Spanish)
        self.month_variations = {
            'jan': 'january', 'feb': 'february', 'mar': 'march',
            'apr': 'april', 'may': 'may', 'jun': 'june',
            'jul': 'july', 'aug': 'august', 'sep': 'september',
            'oct': 'october', 'nov': 'november', 'dec': 'december',
            'ene': 'january', 'feb': 'february', 'mar': 'march',
            'abr': 'april', 'mayo': 'may', 'jun': 'june',
            'jul': 'july', 'ago': 'august', 'sep': 'september',
            'oct': 'october', 'nov': 'november', 'dic': 'december'
        }

    async def handle(self, context: Dict[str, Any]) -> str:
        """Handle date input and validation."""
        try:
            date_input = context.get("user_input", "").strip()
            current_date = datetime.now()
            
            if not date_input:
                return "Please provide a date for your visit."
            
            # Clean the input first
            cleaned_input = self._clean_date_input(date_input)
            print(f"Cleaned input: {cleaned_input}")  # Debug print
            
            # Try parsing the cleaned input
            parsed_date = dateparser.parse(
                cleaned_input,
                settings={
                    'PREFER_DATES_FROM': 'future',
                    'RELATIVE_BASE': current_date,
                    'PREFER_DAY_OF_MONTH': 'first',
                    'DATE_ORDER': 'MDY',
                    'STRICT_PARSING': False
                }
            )
            
            # If parsing fails, try additional patterns
            if not parsed_date:
                # Try extracting just the numeric part for "in X months/years"
                match = re.search(r'in\s*(\d+)\s*(month|year)s?', cleaned_input)
                if match:
                    number = int(match.group(1))
                    unit = match.group(2)
                    if unit == 'month':
                        parsed_date = current_date + relativedelta(months=number)
                    else:
                        parsed_date = current_date + relativedelta(years=number)
            
            # If still no valid date, return error
            if not parsed_date:
                return """
                Sorry, I had trouble understanding that date format.
                Please try again with a format like:
                • "December 2025"
                • "next summer"
                • "in 3 months"
                • "12/2025"
                """
            
            # Check if date is in the past
            if parsed_date < current_date:
                original_date = parsed_date.strftime("%B %Y")
                suggested_date = parsed_date.replace(year=parsed_date.year + 1).strftime("%B %Y")
                return f"""
                ⚠️ Sorry, the date you provided ({original_date}) has already passed.
                
                Please provide a future date for your visit.
                You might want to consider {suggested_date} instead.
                
                For example:
                • "{suggested_date}"
                • "next summer"
                • "in 3 months"
                """
            
            # Format the date consistently
            formatted_date = parsed_date.strftime("%B %Y")
            
            # Use LLM to get seasonal information
            date_analysis = await self.date_chain.ainvoke({
                "date_input": formatted_date,
                "current_date": current_date.strftime("%B %d, %Y")
            })
            
            # Parse the seasonal information
            parts = date_analysis.split("|")
            season = parts[1].strip()
            weather = parts[2].strip()
            tips = parts[3].strip()
            
            # Update state
            self.state.update_state("travel_dates", formatted_date)
            self.state.update_state("season_info", {
                "season": season,
                "weather": weather,
                "tips": tips
            })
            self.state.update_state("current_step", "get_interests")
            
            return f"""
            Great! You're planning to visit in {formatted_date}.

            🌡️ Season Information:
            {season}

            🌤️ Weather Expectations:
            {weather}

            💡 Travel Tips:
            {tips}

            Now, tell me what interests you about Puerto Rico:
            •  Nature and beaches? 🏖️
            •  History and culture? 🏛️
            •  Adventure and sports? 🏄‍♂️
            •  Food and dining? 🍽️
            •  Entertainment? 🎭

            """
                    
        except Exception as e:
            print(f"Error in DateHandler: {str(e)}")
            return """
            Sorry, I had trouble understanding that date format.
            Please try again with a format like:
            • "December 2025"
            • "next summer"
            • "in 3 months"
            """

    def _clean_date_input(self, date_input: str) -> str:
        """Clean and normalize date input from conversational text."""
        # Convert to lowercase and remove extra spaces
        cleaned = date_input.lower().strip()
        
        # Remove punctuation first (commas, periods, etc.)
        cleaned = re.sub(r'[,.!?]', '', cleaned)
        
        # Common conversational patterns to remove
        patterns_to_remove = [
            r'^hi\b\s*',
            r'^hello\b\s*',
            r'^hey\b\s*',
            r'^\s*i\'*\s*am\s*',
            r'^\s*i\'*m\s*',
            r'\s*planning\s*to\s*',
            r'\s*thinking\s*of\s*',
            r'\s*going\s*to\s*',
            r'\s*want\s*to\s*',
            r'\s*would\s*like\s*to\s*',
            r'\s*visiting\s*',
            r'\s*travel\s*',
            r'\s*visit\s*',
            r'\s*there\s*',
            r'\s*around\s*',
            r'\s*about\s*',
            r'\s*maybe\s*',
            r'\s*probably\s*',
            r'\s*in\s+(?=\w+\s+\d{4})',  # Remove 'in' only when followed by month year
            r'^\s*in\s+(?=\d+)',         # Remove 'in' when followed by number
        ]
        
        # Remove each pattern
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, ' ', cleaned, flags=re.IGNORECASE)
        
        # Clean up multiple spaces and trim
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Fix common month spellings
        words = cleaned.split()
        for i, word in enumerate(words):
            if word in self.month_variations:
                words[i] = self.month_variations[word]
        
        # Handle numeric dates (02/03/24 -> February 2024)
        if re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', cleaned):
            try:
                date = datetime.strptime(cleaned, "%m/%d/%y")
                return date.strftime("%B %Y")
            except ValueError:
                try:
                    date = datetime.strptime(cleaned, "%m/%d/%Y")
                    return date.strftime("%B %Y")
                except ValueError:
                    pass
        
        return ' '.join(words)

class ThankYouHandler(BaseHandler):
    """Handler for thank you messages and goodbyes."""
    
    async def handle(self, context: Dict[str, Any]) -> str:
        """Handle gratitude expressions with friendly responses."""
        return """
        You're very welcome! 🤗 It was my pleasure helping you plan your trip to Puerto Rico! 

        🌴 I hope you have an amazing time exploring our beautiful island. Remember to:
        • Check the weather before your trip
        • Keep this itinerary handy
        • Follow local guidelines and respect the environment
        
        ¡Buen viaje! Have a wonderful journey! 🇵🇷 
        """ 