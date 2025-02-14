from typing import Dict
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

QA_PROMPT = PromptTemplate(
    input_variables=[
        "question", "place_name", "metadata", "content", "travel_dates"
    ],
    template="""As a Puerto Rico Travel Assistant, answer this question about \
{place_name}: "{question}"

    Place Information:
    Location: {metadata}
    Description: {content}
    Travel Dates: {travel_dates}
    
    Provide a detailed response that:
    1. Directly answers the question
    2. Includes relevant facts from the description
    3. Adds helpful travel tips based on the location and season
    4. Suggests related places or activities nearby
    5. Maintains a friendly, informative tone

    Format your response with:
    • Clear sections using emojis
    • Practical tips for visitors
    • Relevant seasonal information
    • Any important notes (hours, costs, etc.)

    Remember to:
    - Be specific and accurate with facts
    - Consider the travel season
    - Include local insights
    - Suggest practical tips
    """
)

class PlaceQAChain:
    """Chain for answering questions about specific places."""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def ainvoke(self, inputs: Dict) -> str:
        """Generate a detailed response about a place."""
        try:
            if inputs.get("content"):  # Vector search data available
                prompt = f"""Based on this information about a place in Puerto Rico:
                {inputs['content']}
                
                Answer this question: {inputs['question']}
                
                Include:
                1. Specific details from the content
                2. Location and accessibility
                3. Historical context if relevant
                4. Practical visitor information
                5. Best times to visit
                
                Format with clear sections and emojis.
                """
            else:  # GPT fallback
                prompt = f"""As a Puerto Rico travel expert, answer this question: {inputs['question']}
                
                1. Provide accurate general information
                2. Include historical context
                3. Add practical visitor tips
                4. Format with clear sections
                5. Use emojis for readability
                6. Mention this is based on general knowledge
                """
            
            messages = [
                {"role": "system", "content": "You are a knowledgeable Puerto Rico Travel Assistant."},
                {"role": "user", "content": prompt}
            ]
            
            return await self.llm.ainvoke(messages)
            
        except Exception as e:
            print(f"Error in QA Chain: {str(e)}")
            return "I had trouble generating a response. Please try asking in a different way." 