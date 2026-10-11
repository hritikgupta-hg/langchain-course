import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def main() -> None:
    print("Hello from langchain-course!")
    print(GOOGLE_API_KEY)

    information = """Mukesh Dhirubhai Ambani (born 19 April 1957) is an Indian businessman. He is the chairman and managing director of Reliance Industries, the largest public company in India by market capitalisation in 2025. As of June 2026, he is the richest person in Asia and the 22nd richest in the world, with a net worth of US$91.8 billion.[5][6] He has attracted fame due to his growth and wealth, and criticism for being a plutocrat,[7] and reports of market manipulation, political corruption, cronyism, and exploitation.[8][9][10][11]
    Born in 1957 to Dhirubhai Ambani, the founder of Reliance Industries, and Kokilaben in Aden, Mukesh Ambani completed his studies at St. Xavier's College, Mumbai and Institute of Chemical Technology. He dropped out of Stanford University in 1980 to join Reliance Industries. He expanded the energy ventures of the company, and directed the set up of its largest petroleum refinery at Jamnagar. He took an increasing role in the company after his father developed health issues in the late 1980s. After the death of his father in 2002, the ownership of the companies in the group was divided between him and his younger brother Anil Ambani.[12] Later, he expanded the portfolio of the group and launched or acquired several ventures in energy, finance, healthcare, media, telecommunications, and sports.

    Early life
    Mukesh Dhirubhai Ambani was born on 19 April 1957 in the British crown colony of Aden (present-day Yemen) into a Gujarati Hindu family to Dhirubhai Ambani and Kokilaben Ambani. His father was the founder of Reliance Industries, the largest public company by market capitalization in India.[13] He has a younger brother Anil Ambani and two sisters, Nina Kothari and Dipti Salgaonkar.[14]

    Ambani lived only briefly in Yemen because his father decided to move back to India in 1958 to start a trading business that focused on spices and textiles.[15] The latter was originally named "Vimal" but later changed to "Only Vimal".[16][17] His family lived in a modest two-bedroom apartment in Bhuleshwar, Mumbai, until the 1970s.[18] The family's financial status slightly improved when they moved to India but Ambani still lived in a commune, used public transportation, and never received an allowance.[19] Dhirubhai later purchased a 14-floor apartment block called 'Sea Wind' in Colaba, where Ambani and his brother lived with their families on different floors.[20]

    Education
    Ambani did his primary schooling from the Scindia School, Fort, Gwalior[21] and senior secondary schooling from Hill Grange High School at Peddar Road, Mumbai, along with his brother and Anand Jain, who later became his close associate.[22] After his secondary schooling, he studied at St. Xavier's College, Mumbai.[23] He then received a BE degree in chemical engineering from the Institute of Chemical Technology.[24][25]

    Ambani later enrolled for an MBA at Stanford University (where Steve Ballmer was his classmate[26]) but withdrew in 1980 to help his father build Reliance, which at the time was still a small but fast-growing enterprise.[24] His father felt that real-life skills were harnessed through experiences and not by sitting in a classroom, so he called his son back to India from Stanford to take command of a yarn manufacturing project in his company.[19]

    Ambani was influenced by his teachers William F. Sharpe and Man Mohan Sharma because they are "the kind of professors who made you think out of the box."[24][27][28]

    Career
    Mukesh Ambani set up Reliance Infocomm Limited (later Reliance Communications Limited), which was focused on information and communications technology initiatives.[29] At the age of 24, Ambani was given charge of the construction of Patalganga petrochemical plant when the company was heavily investing in oil refinery and petrochemicals.[30]

    Ambani directed and led the creation of the world's largest grassroots petroleum refinery in Jamnagar, India, which could produce 660,000 barrels per day (33 million tonnes per year) in 2010, integrated with petrochemicals, power generation, port, and related infrastructure.[31] In December 2013, Ambani announced the possibility of a "collaborative venture" with Bharti Airtel in setting up digital infrastructure for the 4G network in India.[32] On 18 June 2014, Ambani said he will invest Rs 1.8 trillion (short scale) across businesses in the next three years and launch 4G broadband services in 2015."""


    summary_template = """
    Given the following information:
    {information}
    Extract the following information:
    1. date of birth
    2. brother's name
    """

    summary_prompt_template = PromptTemplate(
        template=summary_template,
        input_variables=["information"]
    )

    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite",api_key=GOOGLE_API_KEY)
    summary_chain = summary_prompt_template | llm

    summary = summary_chain.invoke({"information": information})

    print(summary.content)

if __name__ == "__main__":
    main()
