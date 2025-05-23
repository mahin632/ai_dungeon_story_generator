import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
model_name = "gpt2"  # You can also use "EleutherAI/gpt-neo-1.3B" for GPT-Neo
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def get_genre_prompt(genre):
    if genre == "Fantasy":
        return "You are a brave knight embarking on an epic quest in a magical land."
    elif genre == "Mystery":
        return "You are a detective investigating a series of strange occurrences in an old town."
    else:
        return "You are an adventurer exploring a mysterious new world."

def generate_multiple_continuations(prompt, num_continuations=3, max_length=200):
    stories = []
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    for _ in range(num_continuations):
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7, top_k=50, top_p=0.9, do_sample=True)
        story = tokenizer.decode(outputs[0], skip_special_tokens=True)
        stories.append(story)
    
    return stories

def save_story_to_file(story, filename="story.txt"):
    with open(filename, "w") as f:
        f.write(story)
    print(f"Story saved to {filename}")

# Streamlit UI
st.title("AI Dungeon Story Generator")
genre = st.selectbox("Choose a Genre", ["Fantasy", "Mystery", "Adventure"])

# Generate prompt based on the selected genre
prompt = get_genre_prompt(genre)

# Input for the user to provide a custom story prompt
user_input = st.text_input("Add your story prompt:", prompt)

# Button to generate story
if st.button("Generate Story"):
    stories = generate_multiple_continuations(user_input)
    
    st.write("### Choose a Story Continuation:")
    for idx, story in enumerate(stories):
        st.write(f"**Option {idx + 1}:**")
        st.write(story)
        st.button(f"Select Option {idx + 1}")
    
    # Option to save the story
    if st.button("Save Story"):
        save_story_to_file(stories[0])  # Save the first story option
        st.write("Story saved successfully!")
