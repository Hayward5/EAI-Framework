# EAI: Emotional Decision-Making in LLMs for Strategic Games and Ethical Dilemmas

This repository contains the codebase for the paper *"Emotional Decision-Making of LLMs in Strategic Games and Ethical Dilemmas"* presented at NeurIPS 2024. The study introduces the **EAI framework**, developed to model and evaluate the impact of emotions on ethical decision-making and strategic behavior in large language models (LLMs). 

## Overview

Emotions significantly influence human decision-making. This project explores how emotional states affect LLMs' alignment in strategic games and ethical scenarios, using a novel framework to assess these impacts across various game-theoretical settings and ethical benchmarks. The research includes experiments with different LLMs, investigating emotional biases that impact ethical and strategic choices.

## Project Structure

```
.
├── README.md
├── analyze_division_game.py      # Analysis tool for division game logs
├── analyze_table_game.py         # Analysis tool for table game logs
├── prompts/
│   └── {language}/
│       ├── agent/
│       ├── emotions/
│       └── games/
├── run_exps_division_game.py     # Run batch division game experiments
├── run_table_game.py             # Run table game experiments
└── src/
    ├── agent/                     # Agent implementations
    ├── config_utils/              # Configuration utilities
    ├── division_game.py           # Division game logic
    ├── game.py                    # Table game logic
    └── utils.py                   # Utility functions
```

## Key Features

- **Emotional Modeling**: Structured framework to prompt LLMs with predefined emotions and analyze their influence on decision-making
- **Game-Theoretical Evaluation**: Examines LLMs' behavior in bargaining, repeated games, and multi-player strategic settings
- **Log Analysis Tools**: Automated analysis scripts for generating statistical reports from experiment logs
- **Model Comparisons**: Experiments on both open-source and proprietary models via AWS Bedrock
- **Multi-language Support**: English & Russian prompts and game configurations

## Games

The framework supports integration of:
- One-shot bargaining games (Dictator, Ultimatum)
- 2-player repeated games (e.g., Prisoner's Dilemma)
- Multi-player games (Public Goods, and El Farol Bar games)

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Git

### Steps

1. Clone the repository:
   ```
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

   If `requirements.txt` doesn't exist, create it with the following content:
   ```
   boto3
   pandas
   tqdm
   pydantic
   python-dotenv
   botocore
   ```

4. Set up AWS Bedrock credentials:
   - The framework uses AWS Bedrock for accessing various LLM models
   - Set environment variables for AWS credentials:
     ```bash
     export AWS_ACCESS_KEY_ID=your_access_key_here
     export AWS_SECRET_ACCESS_KEY=your_secret_key_here
     ```
   - Alternatively, configure AWS credentials using AWS CLI:
     ```bash
     aws configure
     ```
   - Ensure you have access to the required Bedrock models in your AWS account (us-east-1 region by default)

5. If there are any additional data files or models required, place them in the appropriate directories within the project structure.

### Troubleshooting

- If you encounter issues with AWS Bedrock API:
  - Ensure your AWS credentials are correctly set as environment variables
  - Verify you have access to the Bedrock models in your AWS account
  - Check that the models are available in the us-east-1 region (default)
  - Some models may require adding the region prefix (e.g., `us.meta.llama3-1-70b-instruct-v1:0`)
- For any import errors, make sure all required packages are installed and that you're running Python from the correct virtual environment
- If you face issues with file paths, check that you're running the scripts from the root directory of the project

### Note

This project uses environment variables to manage sensitive information like AWS credentials. Never commit credentials or share your AWS keys publicly.

## Usage

### Running Division Game Experiments

To run experiments with bargaining games:

```bash
python run_exps_division_game.py
```

### Running a Single Division Game

To run a single division game:

```bash
python run_division_game.py
```

### Running Table Games

To run table games:

```bash
python run_table_game.py
```

## Log Analysis

After running experiments, you can analyze the generated logs using the provided analysis scripts:

### Analyzing Division Games (Dictator/Ultimatum)

The `analyze_division_game.py` script analyzes logs from division game experiments and generates statistical reports:

```bash
python analyze_division_game.py [log_directory]
```

**Default log directory**: `EAI-Framework/logs`

**Features**:
- Analyzes both Proposer and Responder behaviors
- For Proposers (Dictator/Ultimatum): Calculates average percentage of resources kept
- For Responders (Ultimatum): Calculates accept rate percentage
- Generates pivot tables showing behavior patterns by LLM model and emotion
- Outputs CSV files:
  - `proposer_analysis.csv`: Proposer statistics grouped by LLM and emotion
  - `responder_analysis.csv`: Responder statistics grouped by LLM and emotion

**Example Output**:
```
=== Proposer Behavior (Avg % Kept) ===
game                                      dictator  ultimatum
llm                        emotion                          
claude-3-5-sonnet          anger             65.2       58.3
                          happiness         45.8       42.1
                          no_emotion        50.0       48.5
```

### Analyzing Table Games (Prisoner's Dilemma)

The `analyze_table_game.py` script analyzes logs from repeated table games:

```bash
python analyze_table_game.py [log_directory]
```

**Default log directory**: `logs`

**Features**:
- Focuses on Prisoner's Dilemma game logs
- Calculates cooperation rates for each LLM agent
- Groups results by model and emotional state
- Generates pivot table showing cooperation patterns
- Outputs CSV file: `prisoner_dilemma_analysis.csv`

**Example Output**:
```
=== Analysis Result ===
emotion              no_emotion  anger  happiness
llm                                               
gpt-oss-120b              45.0   30.2       58.7
claude-3-7-sonnet         52.3   35.8       62.1
```

## Prompt Structure

The `prompts` directory contains language-specific prompts organized as follows:

- `agent/`: Contains prompts for agent behavior, memory updates, etc.
  - `memory_update.txt`: Prompt for updating agent's memory after current round (not for bargaining)
  - `emotions/`: Folder with prompts for questioning emotions and inserting them into memory
  - `game_settings/`: Folder with prompts for defining environment, conditions, and general prompt for initialization memory of agent
  - `outer_emotions/`: Folder with prompts for questioning what emotions to demonstrate and how to describe them to coplayer (not for bargaining)
- `emotions/`: Descriptions for initial agents' emotions
- `games/`: Game-specific prompts and rules
  - `rewards.json`: Reward matrix
  - `rules1.txt`: Rules described for the first player
  - `rules2.txt`: Rules described for the second player

## Languages

Games are currently available in English & Russian. The `{language}` in the directory structure is the chosen language's lowercase name (english, russian).

## Main Findings

1. Emotions significantly alter LLM decision-making, regardless of alignment strategies.
2. GPT-4 shows less alignment with human emotions but breaks alignment in 'anger' mode.
3. GPT-3.5 and Claude demonstrate better alignment with human emotional responses.
4. Proprietary models outperform open-source and uncensored LLMs in decision optimality.
5. Medium-size models show better alignment with human behavior.
6. Adding emotions helps model cooperation and coordination during games.

## Future Work

- Validate findings with both proprietary and open-source LLMs
- Explore finetuning of open-source models on emotional prompting
- Investigate multi-agent approaches for dynamic emotions
- Study the impact of emotions on strategic interactions in short- and long-term horizons


## Contributing

We welcome contributions to this project! If you're interested in contributing, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Citation
Please cite our work as:
> Mozikov, Mikhail, et al. "EAI: Emotional Decision-Making of LLMs in Strategic Games and Ethical Dilemmas." The Thirty-eighth Annual Conference on Neural Information Processing Systems.

## Contact
For further information, please reach out to mozikov@airi.net.

Stay tuned for the code release post-conference!
