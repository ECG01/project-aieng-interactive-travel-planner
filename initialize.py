from typing import Dict
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

# Location search chain prompt
LOCATION_SEARCH_PROMPT = PromptTemplate(
    input_variables=["query", "search_type", "location", "specifics"],
    template="""As a Puerto Rico Travel Assistant, search for places based on:

    Query: {query}
    Type: {search_type}
    Location: {location}
    Specifics: {specifics}

    Format each result with:
    1️⃣ **[Place Name]**
    🏷️ [Type of Place]
    📍 [Location], [Direction]
    🔗 [Link if available]
    ℹ️ [Brief description]
    ⏰ [Operating hours if applicable]
    💡 [Special tips or notes]

    Remember to:
    - Include key details about each place
    - Mention any seasonal considerations
    - Add practical visitor information
    - Suggest best times to visit
    - Note any nearby attractions
    """
)

class LocationSearchChain:
    """Chain for searching and formatting location results."""
    
    def __init__(self, llm):
        self.llm = llm
        self.base_chain = LOCATION_SEARCH_PROMPT | llm | StrOutputParser()
    
    async def ainvoke(self, inputs: Dict) -> str:
        """Process search request and format results."""
        try:
            # Get query components
            query = inputs.get("query", "")
            search_type = inputs.get("search_type", "any")
            location = inputs.get("location", "any")
            retriever = inputs.get("retriever")
            
            if not retriever:
                return "Search components not properly initialized."
            
            # Get relevant documents
            docs = await retriever.ainvoke(query)
            
            if not docs:
                return "I couldn't find any places matching those criteria."
            
            # Format results
            formatted_results = []
            for i, doc in enumerate(docs[:3], 1):
                if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                    name = doc.metadata.get('name', 'Unknown location')
                    place_type = doc.metadata.get('type', '')
                    location = doc.metadata.get('town', '')
                    direction = doc.metadata.get('direction', '')
                    
                    result = f"""
                    {i}️⃣ **{name}**
                    🏷️ {place_type}
                    📍 {location}, {direction}
                    🔗 [View Images]()
                    ℹ️ {doc.page_content}
                    💡 Tips: Best to visit during {inputs.get('travel_dates', 'your stay')}
                    """
                    formatted_results.append(result)
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            print(f"Error in LocationSearchChain: {str(e)}")
            return "I encountered an error while searching. Please try again."

async def initialize_components(llm, retriever):
    """Initialize the location search chain and other components."""
    return LocationSearchChain(llm) 