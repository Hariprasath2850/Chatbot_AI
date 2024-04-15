import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

# Load the pre-trained model and tokenizer for translation
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")

# Define language codes
src_lang_code = "en_XX"
tgt_lang_code = "ta_IN"

def translate_to_tamil(input_text):
    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=1024, padding=True, truncation=True)["input_ids"]

    # Generate translated output using the pre-trained model
    translated_output = model.generate(input_ids, max_length=1024, num_beams=4, early_stopping=True, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code])

    # Decode the translated output
    translated_text = tokenizer.batch_decode(translated_output, skip_special_tokens=True)[0]

    return translated_text

def main():
    st.title("English to Tamil Translator")
    st.write("Enter your English text below:")

    # Input text box
    input_text = st.text_area("Input Text")

    # Translate button
    if st.button("Translate"):
        if input_text:
            translation = translate_to_tamil(input_text)
            st.write("Tamil Translation:")
            st.write(translation)
        else:
            st.warning("Please enter some text to translate.")

if __name__ == "__main__":
    main()


