"""100 diverse test strings for Vec2Text evaluation.

All strings are under 32 tokens to ensure optimal Vec2Text performance.
Categories: sentences, questions, instructions, math, compound sentences.
"""

# Category 1: Simple declarative sentences (20)
SENTENCES = [
    "The cat sat on the mat.",
    "Birds fly south for winter.",
    "The sun rises in the east.",
    "Water freezes at zero degrees.",
    "Dogs are loyal companions.",
    "The moon orbits the Earth.",
    "Trees produce oxygen.",
    "Fish swim in the ocean.",
    "The sky is blue today.",
    "Flowers bloom in spring.",
    "Coffee keeps people awake.",
    "Books contain knowledge.",
    "Music soothes the soul.",
    "Time flies when having fun.",
    "Practice makes perfect.",
    "Actions speak louder than words.",
    "The early bird catches the worm.",
    "All that glitters is not gold.",
    "A picture is worth a thousand words.",
    "Knowledge is power.",
]

# Category 2: Questions (20)
QUESTIONS = [
    "What is the capital of France?",
    "How many days are in a year?",
    "Why is the sky blue?",
    "Where do penguins live?",
    "When was the telephone invented?",
    "Who wrote Romeo and Juliet?",
    "What causes earthquakes?",
    "How does photosynthesis work?",
    "Why do leaves change color?",
    "Where is Mount Everest located?",
    "When did World War II end?",
    "Who discovered penicillin?",
    "What is the speed of light?",
    "How far is the moon from Earth?",
    "Why do we dream?",
    "Where do diamonds come from?",
    "When was the internet created?",
    "Who invented the light bulb?",
    "What is the largest ocean?",
    "How do airplanes fly?",
]

# Category 3: Instructions (20)
INSTRUCTIONS = [
    "Calculate the sum of 5 and 3.",
    "Find the square root of 16.",
    "List three primary colors.",
    "Explain how gravity works.",
    "Describe the water cycle.",
    "Compare apples and oranges.",
    "Define the term ecosystem.",
    "Summarize the main idea.",
    "Translate this to French.",
    "Write a short poem about nature.",
    "Solve the equation x plus 2 equals 5.",
    "Draw a triangle with equal sides.",
    "Identify the subject of this sentence.",
    "Count from one to ten.",
    "Name the seven continents.",
    "Spell the word necessary.",
    "Add these numbers together.",
    "Find the missing value in the sequence.",
    "Complete the following pattern.",
    "Arrange these words alphabetically.",
]

# Category 4: Math and numbers (20)
MATH_TEXTS = [
    "The answer is 42.",
    "Two plus two equals four.",
    "The result is approximately 3.14.",
    "X squared equals 25.",
    "The sum is 100.",
    "Half of 50 is 25.",
    "Three times four is twelve.",
    "The average is 7.5.",
    "One third equals 0.333.",
    "The total cost is $49.99.",
    "Pi is approximately 3.14159.",
    "The ratio is 2 to 1.",
    "Subtract 8 from 15 to get 7.",
    "Multiply 6 by 9 to get 54.",
    "Divide 100 by 4 to get 25.",
    "The percentage is 75 percent.",
    "The equation has two solutions.",
    "Zero divided by any number is zero.",
    "Negative five plus ten equals five.",
    "The square of 9 is 81.",
]

# Category 5: Compound sentences (20)
COMPOUND = [
    "If x equals 5, then y is 10.",
    "The train was late, so I missed the meeting.",
    "She likes tea, but he prefers coffee.",
    "Either study hard, or you will fail.",
    "It rained all day, and the streets flooded.",
    "He is smart, yet he makes mistakes.",
    "Work hard, and you will succeed.",
    "I was tired, so I went to bed early.",
    "The movie was long, but it was entertaining.",
    "You can call me, or send an email.",
    "She smiled, for she was truly happy.",
    "I read the book, and I watched the movie.",
    "He ran fast, yet he lost the race.",
    "Study now, or regret it later.",
    "The food was cold, so I heated it up.",
    "I like summer, but I prefer autumn.",
    "She sings well, and she dances beautifully.",
    "It was dark, so I turned on the lights.",
    "He tried hard, yet he failed again.",
    "Take notes, or you will forget.",
]

# Combined list of all 100 sample texts
SAMPLE_TEXTS = SENTENCES + QUESTIONS + INSTRUCTIONS + MATH_TEXTS + COMPOUND

# Category labels for analysis
CATEGORIES = {
    "sentences": (0, 20),
    "questions": (20, 40),
    "instructions": (40, 60),
    "math": (60, 80),
    "compound": (80, 100),
}


def get_category(index: int) -> str:
    """Get category name for a given sample index."""
    for name, (start, end) in CATEGORIES.items():
        if start <= index < end:
            return name
    return "unknown"


if __name__ == "__main__":
    print(f"Total samples: {len(SAMPLE_TEXTS)}")
    for name, (start, end) in CATEGORIES.items():
        print(f"  {name}: {end - start} samples")

    # Verify all are under 32 tokens (rough estimate: words)
    max_words = max(len(s.split()) for s in SAMPLE_TEXTS)
    print(f"\nMax words in any sample: {max_words}")
