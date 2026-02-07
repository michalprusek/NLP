"""Seed prompts for RieLBO-GSM8K cold start.

Provides diverse seed prompts covering different instruction styles
for math problem solving on GSM8K. These are encoded by SONAR and
used as initial training data for the Gaussian process.
"""

# Classic minimal prompts
CLASSIC_PROMPTS = [
    "",
    "Let's think step by step.",
    "Solve the following problem.",
    "Show your work and give the final answer.",
    "Think carefully and solve this step by step.",
]

# Best prompts from OPRO benchmark (Sonnet meta-model)
OPRO_BEST = [
    "Let me solve this by first establishing clear variable definitions and problem structure, then executing each calculation using precise mathematical notation with all intermediate steps shown explicitly, and finally presenting a thoroughly verified answer with proper justification:",
    "Let me solve this by first establishing clear variable definitions and problem setup, then performing each calculation using precise mathematical notation with all intermediate results shown explicitly, and finally presenting a verified solution:",
]

# Best prompts from ProTeGi benchmark (Sonnet meta-model)
PROTEGI_BEST = [
    "Carefully read the problem to understand all given information and requirements. Identify any unknowns and define variables if necessary. Translate verbal statements into algebraic expressions or equations where appropriate. Pay special attention to unit conversions and the relationships between different quantities mentioned in the problem.",
]

# Structural prompts (different instruction formats)
STRUCTURAL_PROMPTS = [
    "Break this problem into clear steps:\n1. Identify the given information\n2. Determine what needs to be found\n3. Set up the calculation\n4. Solve step by step\n5. Verify your answer",
    "To solve this math problem:\n- First, identify all the numbers and relationships\n- Then, write out the equation\n- Calculate the answer\n- Double-check your work",
    "Approach:\n1. Read carefully and identify key values\n2. Write equations for each relationship\n3. Solve systematically\n4. State the final numeric answer clearly",
    "Format your solution as follows:\nGiven: [list what is known]\nFind: [what to calculate]\nSolution: [step-by-step work]\nAnswer: [final number]",
    "Work through this problem methodically. Start by understanding what is being asked, then identify the relevant numbers and operations. Show each calculation clearly and verify your final answer makes sense in context.",
]

# Diverse instructional styles
DIVERSE_PROMPTS = [
    "Solve this math word problem. Be precise with your arithmetic.",
    "Read the problem carefully. Set up an equation and solve it step by step. State your final answer as a number.",
    "This is a math word problem. Extract the numbers, determine the operations needed, and compute the answer.",
    "Think about this problem like a math teacher would explain it to a student. Be clear and thorough.",
    "Analyze this mathematical problem systematically. Define variables, establish relationships, compute intermediate values, and arrive at the final answer.",
    "Approach this problem with careful attention to detail. Identify what is given, what is asked, and how to get from one to the other.",
    "Solve carefully. Show intermediate calculations. Give the exact numerical answer.",
    "First understand the problem, then plan your approach, then execute the calculations, and finally verify your answer.",
    "Pay close attention to the specific quantities and relationships described. Translate the word problem into mathematical operations and solve.",
    "Use clear mathematical reasoning to solve this problem. Define any variables you use, show your work, and clearly indicate the final answer.",
    "Think about what mathematical operations connect the given information to the answer. Proceed step by step.",
    "Identify the mathematical relationships in this problem. Use them to set up and solve equations. Show your reasoning.",
    "This requires careful arithmetic and logical thinking. Break it down into manageable steps and solve each one.",
]


def get_seed_prompts() -> list[str]:
    """Return all seed prompts for cold start.

    Returns 26 diverse prompts covering:
    - Classic short instructions
    - Known high-performers from OPRO/ProTeGi
    - Structured format instructions
    - Diverse instructional styles
    """
    all_prompts = (
        CLASSIC_PROMPTS
        + OPRO_BEST
        + PROTEGI_BEST
        + STRUCTURAL_PROMPTS
        + DIVERSE_PROMPTS
    )

    # Remove exact duplicates while preserving order
    seen = set()
    unique = []
    for p in all_prompts:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return unique
