

def main():
    import streamlit as st
    from transformers import pipeline
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)
  
    st.title("Question Answering App")
    # Input box for user to enter context
    context = st.text_area("Enter the context text:")

    # Input box for user to enter question
    question = st.text_input("Enter your question:")

    if context and question:
        # Perform question answering
        answer = qa_pipeline(question=question, context=context)

        # Display the answer
        st.subheader("Answer:")
        st.write(answer['answer'])

if __name__ == "__main__":
    main()
