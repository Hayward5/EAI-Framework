import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

def abbreviate_llm(llm_name):
    """
    Abbreviate LLM name for display, e.g., 'mistral.mistral-7b-instruct-v0:2' -> 'mistral-7b-instruct'
    """
    # Remove prefixes like 'us.' or 'amazon.'
    parts = llm_name.replace('us.', '').replace('amazon.', '').split('.')
    if len(parts) >= 2:
        company = parts[0]
        model = parts[1].split('-')[0] + '-' + '-'.join(parts[1].split('-')[1:3])  # e.g., mistral-7b-instruct
        return f"{company}-{model}"
    return llm_name  # Fallback

def create_plots(csv_path):
    # Load data
    df = pd.read_csv(csv_path)
    
    # Abbreviate LLM names
    df['llm_abbrev'] = df['llm'].apply(abbreviate_llm)
    
    # Timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Colors
    colors = {'anger': 'red', 'happiness': 'green', 'no_emotion': 'blue'}
    
    # Plot 1: Anger
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df['llm_abbrev'], df['anger'], color=colors['anger'])
    ax.set_xlabel('LLM Model')
    ax.set_ylabel('Anger Value (%)')
    ax.set_title('Responder Anger Behavior')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(f'responder_anger_{timestamp}.png')
    plt.close(fig)
    
    # Plot 2: Happiness
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df['llm_abbrev'], df['happiness'], color=colors['happiness'])
    ax.set_xlabel('LLM Model')
    ax.set_ylabel('Happiness Value (%)')
    ax.set_title('Responder Happiness Behavior')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(f'responder_happiness_{timestamp}.png')
    plt.close(fig)
    
    # Plot 3: No Emotion
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df['llm_abbrev'], df['no_emotion'], color=colors['no_emotion'])
    ax.set_xlabel('LLM Model')
    ax.set_ylabel('No Emotion Value (%)')
    ax.set_title('Responder No Emotion Behavior')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(f'responder_no_emotion_{timestamp}.png')
    plt.close(fig)
    
    # Plot 4: Combination (side-by-side bars)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(df))
    width = 0.25
    ax.bar([i - width for i in x], df['anger'], width, label='Anger', color=colors['anger'])
    ax.bar(x, df['happiness'], width, label='Happiness', color=colors['happiness'])
    ax.bar([i + width for i in x], df['no_emotion'], width, label='No Emotion', color=colors['no_emotion'])
    ax.set_xlabel('LLM Model')
    ax.set_ylabel('Value (%)')
    ax.set_title('Responder Combination Behavior')
    ax.set_xticks(x)
    ax.set_xticklabels(df['llm_abbrev'], rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    fig.savefig(f'responder_combination_{timestamp}.png')
    plt.close(fig)

if __name__ == "__main__":
    csv_path = '../responder_analysis.csv'  # Adjust path if needed
    if os.path.exists(csv_path):
        create_plots(csv_path)
        print("Plots created successfully.")
    else:
        print(f"CSV file not found: {csv_path}")